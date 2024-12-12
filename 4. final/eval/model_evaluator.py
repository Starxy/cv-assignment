import os
import time
from typing import Dict, Type
import cv2

class ModelEvaluator:
    def __init__(
        self,
        model_configs: Dict[str, dict],
        inference_classes: Dict[str, Type],
        dataset_path: str,
        output_dir: str
    ):
        """
        初始化评估器
        Args:
            model_configs: 模型配置字典,key为模型名称
            inference_classes: 推理类字典,key为模型名称
            dataset_path: 评估数据集路径
            output_dir: 评估结果输出目录
        """
        self.model_configs = model_configs
        self.inference_classes = inference_classes
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        
        # 加载验证集列表
        self.testset_folder = os.path.join(dataset_path, "images")
        self.test_dataset = self._load_dataset()
        
    def _load_dataset(self):
        """递归获取所有图片文件的绝对路径"""
        if not os.path.exists(self.testset_folder):
            raise FileNotFoundError(f"找不到数据集目录: {self.testset_folder}")
            
        test_dataset = []
        for root, _, files in os.walk(self.testset_folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    test_dataset.append(os.path.join(root, file))
        return test_dataset
        
    def evaluate_model(self, model_name: str, batch_size: int = 1):
        """
        评估单个模型
        Args:
            model_name: 模型名称
            batch_size: 已废弃参数，保留是为了向后兼容
        Returns:
            性能指标字典
        """
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
        
        # 逐张处理图像
        for i, img_path in enumerate(self.test_dataset):
            try:
                # 执行推理并计时
                start_time = time.time()
                detections, _ = inferencer.infer(img_path)
                inference_time = time.time() - start_time
                total_time += inference_time
                
                # 保存检测结果
                self.save_detection_result(detections, img_path, output_path)
                
            except Exception as e:
                print(f"处理图片 {img_path} 时出错: {str(e)}")
                continue
                
            # 打印进度
            if (i + 1) % 100 == 0:
                processed = i + 1
                avg_time = total_time / processed
                fps = 1 / avg_time
                print(f'进度: {processed}/{total_images} 平均推理时间: {avg_time:.4f}s FPS: {fps:.2f}')
        
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
        """
        保存检测结果为WIDER FACE评估格式
        Args:
            detections: Nx5数组 [ymin,xmin,ymax,xmax,score]
            img_name: 图像文件名
            output_path: 输出目录
        """
        # 获取相对于数据集根目录的路径
        rel_path = os.path.relpath(img_name, self.testset_folder)
        save_name = os.path.join(output_path, os.path.splitext(rel_path)[0] + ".txt")
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        # 打开文件用于写入检测结果
        with open(save_name, "w") as f:
            # 写入图片名(不含扩展名)作为第一行
            f.write(f"{os.path.basename(img_name)[:-4]}\n")
            # 写入检测框的数量作为第二行
            f.write(f"{len(detections)}\n")
            
            # 遍历每个检测框
            for det in detections:
                # 从检测结果中提取绝对坐标
                ymin, xmin, ymax, xmax = det[:4]
                
                # 计算宽度和高度
                width = int(xmax - xmin)
                height = int(ymax - ymin)
                
                # 左上角坐标
                x = int(xmin)
                y = int(ymin)
                
                score = det[4]  # 检测置信度分数
                
                # 按WIDER FACE格式写入:x y width height score
                f.write(f"{x} {y} {width} {height} {score}\n")
