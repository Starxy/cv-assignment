# 轻量级人脸检测模型基准测试

## 实验目标

通过实验对比 BlazeFace、RetinaFace、yolov5face、YuNet 等轻量级人脸检测模型在不同场景下的性能表现，包括准确度、鲁棒性、推理速度和模型大小。

## 测试环境

在 ONNX Runtime 进行推理实验。

通过可公开获取到的 Pytorch 预训练模型和模型定义，使用 pytorch 内置的 onnx 导出为 onnx 格式，然后使用 ONNX Runtime 进行推理。

## 数据集

考虑到端侧人脸检测的实际需求，选择每张图片尽可能只有 1 张人脸的正面图作为验证样本。

通过对样本图进行如下四个方面的数据增强，以评估模型的鲁棒性

- 不同光照条件
- 不同角度
- 遮挡情况
- 模糊图像

## 评价标准

准确度评价标准使用 WIDERFACE Evaluation 标准，评估模型在不同场景下的性能表现。Python 实现参考
- https://github.com/wondervictor/WiderFace-Evaluation
- https://github.com/xiyinmsu/python-wider_eval

推理速度使用每秒推理人脸图片的张数 (FPS) 作为评价标准，横向对比所有模型在 PC 上使用 ONNX Runtime 推理情况下的速度。计算方式为在所有模型使用相同验证集的情况下，计算处理完所有验证集所花费的时间。

模型大小使用模型文件大小 (KB) 作为评价标准，对比所有模型在导出为 onnx 格式后的模型大小。

此外考虑给出一些其他的参考数据，例如说模型的参数量等
