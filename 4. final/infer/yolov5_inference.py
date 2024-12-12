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
        super().__init__(
            model_path=model_path,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold
        )
        self.mean = np.array([0, 0, 0], dtype=np.float32)
        self.std = np.array([1, 1, 1], dtype=np.float32)
        self.input_size = input_size
        
    def preprocess(self, image_path: str):
        """
        图像预处理
        Args:
            image_path: 图片路径
        Returns:
            预处理后的用于推理的数据 (NCHW)
            图片信息
        """

        # 加载并预处理图像
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = original_image.shape[:2]

        # 计算缩放比例
        # 为了保持图像宽高比,取高度和宽度缩放比例的较小值
        # 例如原图640x480,目标尺寸640x640,则r=1.0
        r = min(self.input_size[0] / orig_h, self.input_size[1] / orig_w)

        # 计算padding
        # new_unpad_size 是缩放后的实际尺寸,例如640x480->640x480
        new_unpad_size = (int(round(orig_w * r)), int(round(orig_h * r)))

        # resize图像
        # 如果缩放比例不为1,则需要resize
        # 当r<1时使用INTER_AREA插值(缩小图像),r>1时使用INTER_LINEAR插值(放大图像)
        resized_image = original_image
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            resized_image = cv2.resize(original_image, new_unpad_size, interpolation=interp)

        # letterbox padding
        # dw,dh是需要填充的像素数,例如宽度方向填充(640-480)/2=80个像素
        dw, dh = (self.input_size[1] - new_unpad_size[0]) // 2, (self.input_size[0] - new_unpad_size[1]) // 2
        # 计算上下左右需要填充的像素数
        # 例如640x480的图像,上下各填充80像素变成640x640
        top, bottom = dh, self.input_size[0] - new_unpad_size[1] - dh
        left, right = dw, self.input_size[1] - new_unpad_size[0] - dw
        # 使用固定的灰色值(114,114,114)填充边框
        after_padding_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        after_padding_image = after_padding_image.transpose(2, 0, 1)  # HWC to CHW
        
        # 归一化到0-1
        normalized_image = after_padding_image / 255.0

        # Add batch dimension (1, C, H, W)
        image_info = {
            "height": original_image.shape[0],
            "width": original_image.shape[1],
            "path": image_path,
            "scale": r,
            "pad": (dw, dh)  # 记录padding信息用于后处理
        }
        return np.expand_dims(normalized_image, axis=0).astype(np.float32), image_info
    
    def postprocess(self, ort_outputs, image_info):
        """
        后处理输出结果
        Args:
            ort_outputs: 模型输出
            image_info: 图片原始信息
        Returns:
            检测结果列表，每个元素是 Nx5 的数组:
              [ymin,xmin,ymax,xmax] 表示边界框坐标 使用原始尺寸的绝对坐标
              [score] 表示置信分数
        """
        # 获取预测结果,移除batch维度
        pred = ort_outputs[0].squeeze(0)  # shape=(num_boxes,16)
        
        # 获取原始图像尺寸
        orig_h = image_info['height']
        orig_w = image_info['width']
        
        # 获取缩放和padding参数
        r = image_info['scale']
        dw, dh = image_info['pad']
        
        # 分离输出
        boxes = pred[:, :4]  # x,y,w,h
        scores = pred[:, 4] * pred[:, -1]  # obj_conf * cls_conf
        
        # 置信度过滤
        mask = scores > self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        
        if len(boxes) == 0:
            return np.zeros((0, 5))
        
        # 还原坐标到原始图像
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        
        # 还原坐标，减去padding，除以缩放比例
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = ((x - w/2) - dw) / r  # x1
        boxes_xyxy[:, 1] = ((y - h/2) - dh) / r  # y1 
        boxes_xyxy[:, 2] = ((x + w/2) - dw) / r  # x2
        boxes_xyxy[:, 3] = ((y + h/2) - dh) / r  # y2
        
        # 裁剪到原始图像范围内
        boxes_xyxy[:, [0,2]] = np.clip(boxes_xyxy[:, [0,2]], 0, orig_w)
        boxes_xyxy[:, [1,3]] = np.clip(boxes_xyxy[:, [1,3]], 0, orig_h)
        
        # 组合结果 [ymin,xmin,ymax,xmax,score]
        detections = np.concatenate([
            boxes_xyxy[:, [1,0,3,2]], 
            scores[:, None]
        ], axis=1)
        
        # NMS
        keep = self._nms(boxes_xyxy, scores, self.nms_threshold)
        detections = detections[keep]
        
        return detections
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