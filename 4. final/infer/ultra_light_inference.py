import numpy as np
import cv2
from .base_inference import BaseONNXInference

class UltraLightFaceDetectorONNX(BaseONNXInference):
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.02,
        nms_threshold: float = 0.4,
        input_size: tuple = (320, 320),  # 根据RetinaFace默认设置
    ):
        super().__init__(
            model_path=model_path,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            input_size=input_size,
        )

    def preprocess(self, images):
        """
        图像预处理
        Args:
            images: 单张图片(HWC)或图片列表，RGB格式
        Returns:
            preprocessed_images: NCHW格式的ndarray
        """
        # 确保输入为列表格式
        if not isinstance(images, list):
            images = [images]
        
        processed_images = []
        for img in images:
            # 转换为float32
            img = img.astype(np.float32)
            
            # 直接将图像缩放到目标尺寸
            img = cv2.resize(img, (self.input_size[0], self.input_size[1]), interpolation=cv2.INTER_LINEAR)
            
            # 减均值
            img -= (104, 117, 123)
            
            # HWC转CHW
            img = img.transpose(2, 0, 1)
            processed_images.append(img)
        
        # 堆叠为batch
        batch_images = np.stack(processed_images, axis=0)
        return batch_images

    def postprocess(self, ort_outputs, image_info):
        """
        后处理模型输出
        Args:
            ort_outputs: 模型原始输出(loc, conf, landms)
            image_info: 包含原始图像信息的字典列表
        Returns:
            list of Nx5 arrays, 每个数组包含[ymin,xmin,ymax,xmax,score]
        """
        # 解析模型输出
        loc, conf, _ = ort_outputs
        batch_size = loc.shape[0]
        results = []
        
        # 对每张图片进行处理
        for batch_idx in range(batch_size):
            # 获取当前图片的输出
            cur_loc = loc[batch_idx]
            cur_conf = conf[batch_idx]
            
            # 获取得分
            scores = cur_conf[:, 1]
            
            # 过滤低置信度的检测框
            mask = scores > self.conf_threshold
            boxes = cur_loc[mask]
            scores = scores[mask]
            
            if len(boxes) == 0:
                results.append(np.zeros((0, 5)))
                continue
            
            # 组合boxes和scores
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            
            # 执行NMS
            keep = self._py_cpu_nms(dets, self.nms_threshold)
            dets = dets[keep]
            
            # 转换为相对坐标(0-1)
            h, w = image_info[batch_idx]['height'], image_info[batch_idx]['width']
            dets[:, 0] /= w  # x1
            dets[:, 1] /= h  # y1
            dets[:, 2] /= w  # x2
            dets[:, 3] /= h  # y2
            
            # 转换为[ymin,xmin,ymax,xmax,score]格式
            result = np.zeros_like(dets)
            result[:, 0] = dets[:, 1]  # ymin
            result[:, 1] = dets[:, 0]  # xmin
            result[:, 2] = dets[:, 3]  # ymax
            result[:, 3] = dets[:, 2]  # xmax
            result[:, 4] = dets[:, 4]  # score
            
            results.append(result)
        
        return results

    @staticmethod
    def _py_cpu_nms(dets, thresh):
        """
        纯Python实现的NMS
        Args:
            dets: nx5数组，每行为[x1,y1,x2,y2,score]
            thresh: NMS阈值
        Returns:
            保留的检测框索引
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        
        return keep