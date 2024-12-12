from model_evaluator import ModelEvaluator
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infer.retinaface_inference import RetinaFaceInference
from infer.yolov5_inference import YOLOV5FaceInference
from infer.blazeface_128_inference import BlazeFace128Inference

def main():
    # 模型配置
    model_configs = {
        "retinaface_mv1_0.25": {
            "model_path": "models/retainface/retinaface_mv1_0.25.onnx",
            "conf_threshold": 0.02,
            "nms_threshold": 0.4
        },
        "retinaface_mv2": {
            "model_path": "models/retainface/retinaface_mv2.onnx",
            "conf_threshold": 0.02,
            "nms_threshold": 0.4
        },
        # "yolov5n_0.5_face": {
        #     "model_path": "models/yolov5face/yolov5n-0.5.onnx",
        #     "conf_threshold": 0.02,
        #     "nms_threshold": 0.4
        # },
        # "yolov5n_face": {
        #     "model_path": "models/yolov5face/yolov5n-face.onnx",
        #     "conf_threshold": 0.02,
        #     "nms_threshold": 0.4
        # },
        # "blazeface": {
        #     "model_path": "models/blaze_face/blazeface_128.onnx",
        #     "anchors_path": "models/blaze_face/anchors_128.npy",
        #     "conf_threshold": 0.75,
        #     "nms_threshold": 0.3,
        #     "input_size": (128, 128)
        # }
    }
    
    # 推理类映射
    inference_classes = {
        "retinaface_mv1_0.25": RetinaFaceInference,
        "retinaface_mv2": RetinaFaceInference,
        "yolov5n_0.5_face": YOLOV5FaceInference,
        "yolov5n_face": YOLOV5FaceInference,
        "blazeface_128": BlazeFace128Inference
    }
    
    # 初始化评估器
    evaluator = ModelEvaluator(
        model_configs=model_configs,
        inference_classes=inference_classes,
        dataset_path="C:\\Project\\ai\\cv-assignment\\4. final\\dataset\\wilderface_val",
        output_dir="C:\\Project\\ai\\cv-assignment\\4. final\\eval\\results"
    )
    
    # 评估所有模型
    results = {}
    for model_name in model_configs.keys():
        print(f"\n开始评估模型: {model_name}")
        results[model_name] = evaluator.evaluate_model(model_name)
        
    # 输出评估结果
    print("\n评估结果汇总:")
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"FPS: {result['fps']:.2f}")
        print(f"总图片数: {result['total_images']}")
        print(f"总耗时: {result['total_time']:.2f}秒")
        print(f"平均推理时间: {result['avg_time']*1000:.2f}ms")

if __name__ == "__main__":
    main() 