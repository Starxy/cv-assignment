from model_evaluator import ModelEvaluator
from onnx.retinaface_inference import RetinaFaceONNXInference
# 导入其他模型的推理类

def main():
    # 模型配置
    model_configs = {
        "retinaface": {
            "model_path": "../models/retainface/retinaface_mv2.onnx",
            "model_name": "mobilenetv2",
            "conf_threshold": 0.02,
            "nms_threshold": 0.4
        },
        # 添加其他模型配置
    }
    
    # 推理类映射
    inference_classes = {
        "retinaface": RetinaFaceONNXInference,
        # 添加其他模型的推理类
    }
    
    # 初始化评估器
    evaluator = ModelEvaluator(
        model_configs=model_configs,
        inference_classes=inference_classes,
        dataset_path="../dataset/widerface/val",
        output_dir="../eval/results"
    )
    
    # 评估所有模型
    results = {}
    for model_name in model_configs.keys():
        print(f"评估模型: {model_name}")
        results[model_name] = evaluator.evaluate_model(model_name)
        
    # 输出评估结果
    for model_name, result in results.items():
        print(f"\n{model_name} 评估结果:")
        print(f"FPS: {result['fps']:.2f}")
        print(f"总图片数: {result['total_images']}")
        print(f"总耗时: {result['total_time']:.2f}秒")

if __name__ == "__main__":
    main() 