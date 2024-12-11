import numpy as np
import cv2
from .base_inference import BaseONNXInference
from models.retainface.priors import generate_priors, decode_boxes, decode_landmarks, RETINAFACE_CONFIG

class RetinaFaceInference(BaseONNXInference):
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.02,
        nms_threshold: float = 0.4,
        input_size: tuple = (640, 640),
        rgb_means: tuple = (104, 117, 123),
        prior_config: dict = None
    ):
        """
        初始化RetinaFace ONNX推理类
        Args:
            model_path: ONNX模型路径
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
            input_size: 输入尺寸
            rgb_means: RGB均值，用于图像归一化
            prior_config: 先验框配置，如果为None则使用默认配置
        """
        super().__init__(
            model_path=model_path,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            input_size=input_size
        )
        self.rgb_means = rgb_means
        self.prior_config = prior_config if prior_config is not None else RETINAFACE_CONFIG
        # 预生成先验框以提高效率
        self.priors = generate_priors(self.input_size, self.prior_config)
    
    def preprocess(self, images):
        """
        图像预处理
        Args:
            images: 单张图片或图片列表
        Returns:
            预处理后的批量数据(NCHW)
        """
        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            images = [images]
            
        processed_images = []
        for img in images:
            # 调整图像大小到模型输入尺寸
            if img.shape[:2] != self.input_size:
                img = cv2.resize(img, self.input_size[::-1])  # OpenCV使用(width, height)格式
            
            # 减去均值
            img = img.astype(np.float32)
            img -= self.rgb_means
            # HWC to CHW
            img = img.transpose(2, 0, 1)
            processed_images.append(img)
            
        batch_input = np.stack(processed_images, axis=0)
        return batch_input
    
    def _nms(self, dets):
        """非极大值抑制"""
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
            inds = np.where(ovr <= self.nms_threshold)[0]
            order = order[inds + 1]
        return keep

    def _scale_boxes_landmarks(self, boxes, landmarks, image_info):
        """将预测框和关键点缩放到原始图像尺寸"""
        height, width = image_info['height'], image_info['width']
        scale_x = width / self.input_size[1]
        scale_y = height / self.input_size[0]
        
        # 缩放边界框
        if boxes.size > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            
        # 缩放关键点
        if landmarks.size > 0:
            landmarks[:, 0::2] *= scale_x
            landmarks[:, 1::2] *= scale_y
            
        return boxes, landmarks

    def postprocess(self, ort_outputs, image_info):
        """
        后处理输出结果
        Args:
            ort_outputs: 模型输出 [loc, conf, landmarks]
            image_info: 图像信息列表
        Returns:
            检测结果列表，每个元素是 Nx5 的数组:
              [ymin,xmin,ymax,xmax] 表示边界框坐标 (相对坐标,范围0-1)
              [score] 表示置信分数
        """
        # 解包模型输出
        loc, conf, landmarks = ort_outputs
        batch_size = loc.shape[0]
        results = []

        for batch_idx in range(batch_size):
            # 获取当前批次的输出
            batch_loc = loc[batch_idx]
            batch_conf = conf[batch_idx]

            batch_landmarks = landmarks[batch_idx]
            
            # 解码boxes和landmarks
            boxes = decode_boxes(
                batch_loc, 
                self.priors, 
                self.prior_config['variance']
            )
            landmarks = decode_landmarks(
                batch_landmarks, 
                self.priors, 
                self.prior_config['variance']
            )
            
            # 获取置信度分数
            scores = batch_conf[:, 1]
            
            # 应用置信度阈值
            mask = scores > self.conf_threshold
            boxes = boxes[mask]
            landmarks = landmarks[mask]
            scores = scores[mask]
            
            # 组合检测结果
            detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            
            # 应用NMS
            keep = self._nms(detections)
            
            # 保存最终结果
            if len(keep) > 0:
                detections = detections[keep]
                # 将坐标缩放回原始图像尺寸
                # detections[:, :4], _ = self._scale_boxes_landmarks(
                #     detections[:, :4], 
                #     landmarks[keep], 
                #     image_info[batch_idx]
                # )
                # 调整坐标顺序为[ymin,xmin,ymax,xmax]
                result = np.concatenate([
                    detections[:, 1:2],  # ymin
                    detections[:, 0:1],  # xmin 
                    detections[:, 3:4],  # ymax
                    detections[:, 2:3],  # xmax
                    detections[:, 4:5]   # score
                ], axis=1)
            else:
                result = np.zeros((0, 5))  # 5 = 4(边界框) + 1(置信度)
                
            results.append(result)
            
        return results if len(results) > 1 else results[0] 