# 血细胞目标检测实验

本实验旨在比较 YOLOv3 和 SSD 两种经典目标检测算法在血细胞检测任务上的性能表现。

## 数据集

使用 [Blood Cell Object Detection](https://huggingface.co/datasets/keremberke/blood-cell-object-detection) 数据集:

- **类别**: 4种血细胞类型(中性粒细胞、嗜酸性粒细胞、淋巴细胞和单核细胞)
- **数据规模**:
  - 训练集: 364张
  - 验证集: 60张  
  - 测试集: 100张
- **标注格式**: COCO格式

## 实验环境

### 依赖

bash
torch>=1.7.0
torchvision>=0.8.1
opencv-python
numpy
pycocotools

## 实验设置

### 数据预处理
- 数据增强策略:
  - 随机亮度对比度调整
  - 水平翻转
  - 随机90度旋转
  - 标准化
  
### 模型配置

#### YOLOv3
- Backbone: Darknet-53
- 输入尺寸: 416×416
- Batch size: 16
- 学习率: 1e-3
- 优化器: Adam

#### SSD
- Backbone: VGG16
- 输入尺寸: 300×300
- Batch size: 32
- 学习率: 1e-3
- 优化器: SGD

### 训练策略
- 训练轮数: 100 epochs
- 学习率调整: CosineAnnealingLR
- 早停策略: 验证集mAP连续10轮无提升

## 评估指标

- mAP (mean Average Precision)
- AP@.5 (IoU=0.5)
- AP@.75 (IoU=0.75)
- 检测速度(FPS)
- 模型大小

## 实验结果

### 性能对比

| 模型    | mAP  | AP@.5 | AP@.75 | FPS | 模型大小(MB) |
|---------|------|-------|--------|-----|--------------|
| YOLOv3  | -    | -     | -      | -   | -           |
| SSD     | -    | -     | -      | -   | -           |

### 可视化结果

*待补充检测结果可视化图像*

## 消融实验

1. Backbone影响
2. 数据增强策略影响
3. 输入分辨率影响

## 结论与分析

*待实验完成后补充*

## 使用说明

### 1. 环境配置

