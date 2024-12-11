import cv2
import numpy as np
from .base_inference import BaseONNXInference

class YOLOV5FaceInference(BaseONNXInference):
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.02,
        nms_threshold: float = 0.4,
        input_size: tuple = (640, 640),
    ):
        super().__init__(model_path, conf_threshold, nms_threshold, input_size)
        self.mean = np.array([0, 0, 0], dtype=np.float32)
        self.std = np.array([1, 1, 1], dtype=np.float32)
        
    def preprocess(self, images):
        """预处理RGB格式图像
        Args:
            images: 单张图片(HWC)或图片列表([HWC,...])
            target_size: 目标尺寸 (height, width)
        Returns:
            input_tensor: 预处理后的输入tensor, shape=(N,3,h,w), dtype=np.float32
        """
        # 统一输入格式为列表
        if isinstance(images, np.ndarray):
            images = [images]
        
        # 存储预处理后的图片
        processed_imgs = []
        
        for img in images:
            # 保存原始图像尺寸
            orig_h, orig_w = img.shape[:2]
            
            # 计算缩放比例
            r = min(self.input_size[0] / orig_h, self.input_size[1] / orig_w)
            
            # 计算padding
            new_unpad = (int(round(orig_w * r)), int(round(orig_h * r)))
            dw, dh = (self.input_size[1] - new_unpad[0]) // 2, (self.input_size[0] - new_unpad[1]) // 2
            
            # resize图像
            if r != 1:
                interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
                img = cv2.resize(img, new_unpad, interpolation=interp)
            
            # letterbox padding
            top, bottom = dh, self.input_size[0] - new_unpad[1] - dh
            left, right = dw, self.input_size[1] - new_unpad[0] - dw
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            
            # 预处理
            img = img.transpose(2, 0, 1)  # HWC to CHW
            img = np.ascontiguousarray(img, dtype=np.float32)  # contiguous array memory
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            
            processed_imgs.append(img)
        
        # 堆叠成batch
        input_tensor = np.stack(processed_imgs, axis=0)
        
        return input_tensor
    
    def postprocess(self, ort_outputs, image_info):
        """后处理模型输出"""
        output = ort_outputs[0]  # shape=(N,num_boxes,16)
        batch_size = output.shape[0]
        
        batch_results = []
        
        for i in range(batch_size):
            # 获取原始图像尺寸
            orig_h = image_info[i]['height']
            orig_w = image_info[i]['width']
            
            # 计算缩放和padding参数
            r = min(self.input_size[0]/orig_h, self.input_size[1]/orig_w)
            new_unpad_h = int(round(orig_h * r))
            new_unpad_w = int(round(orig_w * r))
            dw = (self.input_size[1] - new_unpad_w) // 2
            dh = (self.input_size[0] - new_unpad_h) // 2
            
            # 获取预测结果
            pred = output[i]
            
            # 分离输出
            boxes = pred[:, :4]  # x,y,w,h
            scores = pred[:, 4] * pred[:, -1]  # obj_conf * cls_conf
            
            # 置信度过滤
            mask = scores > self.conf_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            
            if len(boxes) == 0:
                batch_results.append(np.zeros((0, 5)))
                continue
            
            # 还原坐标到原始图像
            x = boxes[:, 0]
            y = boxes[:, 1]
            w = boxes[:, 2]
            h = boxes[:, 3]
            
            # 还原坐标，减去padding，除以缩放比例
            boxes_xyxy = np.zeros_like(boxes)
            boxes_xyxy[:, 0] = ((x - w/2) - dw) / r / orig_w  # x1
            boxes_xyxy[:, 1] = ((y - h/2) - dh) / r / orig_h  # y1 
            boxes_xyxy[:, 2] = ((x + w/2) - dw) / r / orig_w  # x2
            boxes_xyxy[:, 3] = ((y + h/2) - dh) / r / orig_h  # y2
            
            # 裁剪到[0,1]范围
            boxes_xyxy = np.clip(boxes_xyxy, 0, 1)
            
            # 组合结果 [ymin,xmin,ymax,xmax,score]
            detections = np.concatenate([
                boxes_xyxy[:, [1,0,3,2]], 
                scores[:, None]
            ], axis=1)
            
            # NMS
            keep = self._nms(boxes_xyxy, scores, self.nms_threshold)
            detections = detections[keep]
            
            batch_results.append(detections)
        
        return batch_results
    
    def _nms(self, boxes, scores, iou_thres):
        """非极大值抑制"""
        y1 = boxes[:, 0]
        x1 = boxes[:, 1]
        y2 = boxes[:, 2]
        x2 = boxes[:, 3]
        areas = (y2 - y1) * (x2 - x1)
        
        # 按分数排序
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # 计算IoU
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]
            
        return keep