import torch
from torch import nn
from torch.nn import functional as F
import math

def cls_predictor(num_inputs, num_anchors, num_classes):
    """类别预测层"""
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)

def bbox_predictor(num_inputs, num_anchors):
    """边界框预测层"""
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

def flatten_pred(pred):
    """将预测结果转换为二维张量"""
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    """连结多尺度的预测结果"""
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

def down_sample_blk(in_channels, out_channels):
    """高和宽减半块"""
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                            kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

def base_net():
    """基础网络块"""
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

def get_blk(i):
    """获取第i个块"""
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

def multibox_prior(feature_map, sizes=[0.75, 0.5], ratios=[1, 2, 0.5]):
    """生成以每个像素为中心具有不同尺寸和宽高比的锚框
    
    Args:
        feature_map: 输入特征图, shape (N, C, H, W)
        sizes: 锚框大小列表, 相对于特征图的比例
        ratios: 锚框的宽高比列表
    Returns:
        anchors: 所有锚框的坐标, shape (1, H * W * num_anchors, 4)
    """
    pairs = [] # 存储不同大小和比例组合的宽和高
    for r in ratios:
        pairs.append([sizes[0] * math.sqrt(r), sizes[0] / math.sqrt(r)])
    for s in sizes[1:]:
        pairs.append([s, s])
    pairs = torch.tensor(pairs)
    
    _, _, h, w = feature_map.shape
    # 每个中心点的相对坐标
    steps_h = 1.0 / h  
    steps_w = 1.0 / w
    
    # 生成所有像素的中心点坐标
    center_h = (torch.arange(h) + 0.5) * steps_h
    center_w = (torch.arange(w) + 0.5) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    
    # 扩展中心点坐标为锚框坐标
    # 左上角和右下角的坐标计算
    box_w, box_h = pairs[:, 0], pairs[:, 1]
    boxes = torch.stack((
        shift_x.unsqueeze(1) - 0.5 * box_w.unsqueeze(0),  # xmin
        shift_y.unsqueeze(1) - 0.5 * box_h.unsqueeze(0),  # ymin
        shift_x.unsqueeze(1) + 0.5 * box_w.unsqueeze(0),  # xmax
        shift_y.unsqueeze(1) + 0.5 * box_h.unsqueeze(0)   # ymax
    ), dim=2)
    
    boxes = boxes.clamp(0, 1)  # 将坐标限制在[0,1]范围内
    return boxes.unsqueeze(0)  # 添加batch维度

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    """块的前向计算"""
    Y = blk(X)
    anchors = multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        
        # 定义每个块的输入通道数
        idx_to_in_channels = [64, 128, 128, 128, 128]
        
        # 锚框的大小和宽高比
        self.sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], 
                     [0.71, 0.79], [0.88, 0.961]]
        self.ratios = [[1, 2, 0.5]] * 5
        self.num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1
        
        # 创建各个块
        for i in range(5):
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', 
                   cls_predictor(idx_to_in_channels[i], self.num_anchors, num_classes))
            setattr(self, f'bbox_{i}', 
                   bbox_predictor(idx_to_in_channels[i], self.num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        
        # 对每个块进行前向计算
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), 
                self.sizes[i], self.ratios[i],
                getattr(self, f'cls_{i}'), 
                getattr(self, f'bbox_{i}'))

        # 将预测结果连结
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        
        return anchors, cls_preds, bbox_preds

def focal_loss(gamma, x):
    """Focal loss"""
    return -(1 - x) ** gamma * torch.log(x)
