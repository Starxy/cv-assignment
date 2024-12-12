from abc import ABC, abstractmethod
import numpy as np
import onnxruntime as ort
import cv2

class BaseONNXInference(ABC):
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.02,
        nms_threshold: float = 0.4,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.ort_session = ort.InferenceSession(model_path)
    
    @abstractmethod
    def preprocess(self, image_path: str):
        """
        图像预处理
        Args:
            image_path: 图片路径
        Returns:
            预处理后的用于推理的数据 (NCHW)
            图片信息
        """
        pass
    
    @abstractmethod
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
        pass
    
    def infer(self, image_path: str):
        """
        推理流程
        Args:
            image_path: 图片路径
        Returns:
            tuple: (检测结果, 图片路径)
                - 检测结果: Nx5 的检测结果数组，包含边界框坐标和置信分数，使用原始尺寸的绝对坐标
                - 图片路径: 图片路径
        """
        
        # 预处理
        input_tensor, image_info = self.preprocess(image_path)
        
        # 使用 ONNX Runtime 执行推理
        ort_outputs = self.ort_session.run(None, {'input': input_tensor})
        
        # 对推理结果进行后处理
        detections = self.postprocess(ort_outputs, image_info)

        return detections, image_path