import os
import time
from typing import Dict, Type
from tqdm import tqdm
import cv2
import numpy as np

class ModelEvaluator:
    def __init__(
        self,
        model_configs: Dict[str, dict],
        inference_classes: Dict[str, Type],
        dataset_path: str,
        output_dir: str
    ):
        self.model_configs = model_configs
        self.inference_classes = inference_classes
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        
        # 加载数据集列表
        self.testset_folder = os.path.join(dataset_path, "images")
        self.testset_list = os.path.join(dataset_path, "wider_val.txt")
        self.test_dataset = self._load_dataset()
        
    def _load_dataset(self):
        """加载WIDERFACE验证集列表"""
        if not os.path.exists(self.testset_list):
            raise FileNotFoundError(f"找不到数据集列表文件: {self.testset_list}")
        with open(self.testset_list, 'r') as fr:
            test_dataset = fr.read().split()
        return test_dataset
        
    def evaluate_model(self, model_name: str):
        """评估单个模型"""
        config = self.model_configs[model_name]
        inference_class = self.inference_classes[model_name]
        
        # 初始化推理器
        inferencer = inference_class(**config)
        
        # 准备输出目录
        output_path = os.path.join(self.output_dir, model_name)
        os.makedirs(output_path, exist_ok=True)
        
        # 记录推理时间
        total_time = 0
        total_images = len(self.test_dataset)
        
        # 遍历数据集
        for i, img_name in enumerate(tqdm(self.test_dataset, desc=f"评估 {model_name}")):
            image_path = os.path.join(self.testset_folder, img_name)
            
            try:
                # 执行推理并计时
                start_time = time.time()
                detections, _ = inferencer.infer(image_path)
                inference_time = time.time() - start_time
                total_time += inference_time
                
                # 保存检测结果
                self.save_detection_result(detections, img_name, output_path)
                
            except Exception as e:
                print(f"处理图像 {image_path} 时出错: {str(e)}")
                continue
                
            # 打印进度
            if (i + 1) % 100 == 0:
                avg_time = total_time / (i + 1)
                fps = 1 / avg_time
                print(f'进度: {i + 1}/{total_images} 平均推理时间: {avg_time:.4f}s FPS: {fps:.2f}')
        
        # 计算最终性能指标
        avg_time = total_time / total_images
        fps = 1 / avg_time if avg_time > 0 else 0
        
        return {
            "fps": fps,
            "total_images": total_images,
            "total_time": total_time,
            "avg_time": avg_time
        }
    
    def save_detection_result(self, detections, img_name, output_path):
        """保存检测结果为WIDERFACE评估格式的txt文件"""
        # 创建保存路径
        save_name = os.path.join(output_path, img_name[:-4] + ".txt")
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
            
        # 写入检测结果
        with open(save_name, "w") as f:
            # 写入图像名和检测框数量
            f.write(f"{os.path.basename(img_name)[:-4]}\n")
            f.write(f"{len(detections)}\n")
            
            # 写入每个检测框的信息
            for det in detections:
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, det[:4])
                # 计算宽高
                w = x2 - x1
                h = y2 - y1
                # 获取置信度
                score = det[4]
                # 写入格式: x y w h score
                f.write(f"{x1} {y1} {w} {h} {score}\n")
    
    def get_dataset_images(self):
        """获取数据集图像路径列表"""
        return [os.path.join(self.testset_folder, img_name) for img_name in self.test_dataset]