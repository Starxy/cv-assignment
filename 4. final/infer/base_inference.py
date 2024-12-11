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
        input_size: tuple = (640, 640),
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.ort_session = ort.InferenceSession(model_path)
    
    @abstractmethod
    def preprocess(self, images):
        """
        图像预处理，支持单张图片或批量图片
        Args:
            images: 单张图片 (HWC) 或图片列表或批量图片 (NHWC)
        Returns:
            预处理后的批量数据 (NCHW)
        """
        pass
    
    @abstractmethod
    def postprocess(self, ort_outputs, image_info):
        """
        后处理输出结果
        Args:
            ort_outputs: 模型输出
            image_info: 单张或多张图片的信息
        Returns:
            检测结果列表，每个元素是 Nx5 的数组:
              [ymin,xmin,ymax,xmax] 表示边界框坐标 (相对坐标,范围 0-1)
              [score] 表示置信分数
        """
        pass

    def infer(self, images, batch_size: int = 30):
        """
        推理流程
        Args:
            images: 图片路径、numpy 数组或其列表
            batch_size: 批处理大小
        Returns:
            tuple: (检测结果, 原始图像, 图片路径)
                - 检测结果: 单个检测结果或检测结果列表
                - 原始图像: 单张图片或图片列表
                - 图片路径: 单个路径或路径列表，如果输入是 numpy 数组则对应位置为 None
        """
        # 统一输入格式
        if isinstance(images, (str, np.ndarray)):
            images = [images]
            
        # 处理图像和收集信息
        processed_images = []
        image_info_list = []
        original_images = []
        image_paths = []
        
        for img in images:
            if isinstance(img, str):
                original_image = cv2.imread(img)
                if original_image is None:
                    raise ValueError(f"无法读取图像: {img}")
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                img_path = img
            else:
                original_image = img
                img_path = None
            
            original_images.append(original_image)
            image_paths.append(img_path)
            image_info_list.append({
                "height": original_image.shape[0],
                "width": original_image.shape[1],
                "path": img_path
            })
            processed_images.append(original_image)
        
        # 批量预处理
        input_tensor = self.preprocess(processed_images)
        
        # 批量推理
        all_detections = []
        
        # 按批次处理图片
        # 按批次循环处理所有图片
        for i in range(0, len(processed_images), batch_size):
            # 获取当前批次的输入张量,处理最后一组不足 batch_size 的情况
            end_idx = min(i + batch_size, len(processed_images))
            batch_input = input_tensor[i:end_idx]
            # 使用 ONNX Runtime 执行推理
            ort_outputs = self.ort_session.run(None, {'input': batch_input})
            # 获取当前批次图片的信息
            batch_info = image_info_list[i:end_idx]
            # 对推理结果进行后处理
            detections = self.postprocess(ort_outputs, batch_info)
            # 将检测结果添加到总结果列表中，如果不是列表则转换为列表
            all_detections.extend(detections if isinstance(detections, list) else [detections])
        # 如果只有一张图片,返回单个结果而不是列表
        if len(images) == 1:
            return all_detections[0], original_images[0], image_paths[0]
        return all_detections, original_images, image_paths