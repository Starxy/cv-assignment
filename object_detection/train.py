"""
SSD目标检测模型训练模块

@Author: DONG Jixing
@Date: 2024-11-24
"""

import torch
from torch import nn
from ssd import TinySSD
from utils import Timer, Animator, Accumulator
import os
from datetime import datetime

def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2的形状:
    # boxes1：(boxes1的数量,4),
    # boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量,),
    # areas2：(boxes2的数量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # inter_upperlefts,inter_lowerrights,inters的形状:
    # (boxes1的数量,boxes2的数量,2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # 根据阈值，决定是否分配真实边界框
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map

def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        # 过滤掉无效的标签（类别为-1的标签）
        valid_labels = label[label[:, 0] >= 0]
        
        if len(valid_labels) == 0:
            # 如果没有有效标签，将所有锚框标记为背景
            class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
            assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
            bbox_mask = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        else:
            # 只使用有效标签进行分配
            anchors_bbox_map = assign_anchor_to_bbox(
                valid_labels[:, 1:], anchors, device)
            bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
                1, 4)
            # 将类标签和分配的边界框坐标初始化为零
            class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                       device=device)
            assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                      device=device)
            # 使用真实边界框来标记锚框的类别。
            # 如果一个锚框没有被分配，标记其为背景（值为零）
            indices_true = torch.nonzero(anchors_bbox_map >= 0)
            bb_idx = anchors_bbox_map[indices_true]
            class_labels[indices_true] = valid_labels[bb_idx, 0].long() + 1
            assigned_bb[indices_true] = valid_labels[bb_idx, 1:]
            # 偏移量转换
            offset = offset_boxes(anchors, assigned_bb) * bbox_mask
            batch_offset.append(offset.reshape(-1))
            batch_mask.append(bbox_mask.reshape(-1))
            batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss = nn.L1Loss(reduction='none')
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

def evaluate_accuracy(net, data_iter, device):
    """评估模型的分类和边界框预测准确性"""
    net.eval()
    metric = Accumulator(4)  # 分类正确数、分类总数、bbox误差和、bbox总数
    
    with torch.no_grad():
        for features, target in data_iter:
            X, Y = features.to(device), target.to(device)
            # 获取预测结果
            anchors, cls_preds, bbox_preds = net(X)
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            
            # 评估指标
            cls_acc = float((cls_preds.argmax(dim=-1).type(
                cls_labels.dtype) == cls_labels).sum())
            bbox_mae = float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
            
            metric.add(cls_acc, cls_labels.numel(),
                      bbox_mae, bbox_labels.numel())
    
    cls_err = 1 - metric[0] / metric[1]
    bbox_mae = metric[2] / metric[3]
    
    return cls_err, bbox_mae

def train(net, train_iter, val_iter, num_epochs, lr, device, save_model=True, title="SSD Training"):
    """
    训练SSD模型的主函数
    Args:
        save_model (bool): 是否保存模型，默认为True
    """
    print(f'training on {device}')
    net.to(device)
    
    # 定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=5e-4)
    
    # 创建动画器
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                       legend=['train cls error', 'train bbox mae',
                              'val cls error', 'val bbox mae'],
                       title=title)
    
    timer = Timer()
    for epoch in range(num_epochs):
        # 训练阶段
        metric = Accumulator(4)
        net.train()
        
        for features, target in train_iter:
            timer.start()
            optimizer.zero_grad()
            X, Y = features.to(device), target.to(device)
            
            # 获取锚框、类别预测和边界框预测
            anchors, cls_preds, bbox_preds = net(X)

            # 获取真实边界框标签和掩码
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            
            # 计算损失并反向传播
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.mean().backward()
            optimizer.step()
            
            # 评估指标
            cls_acc = float((cls_preds.argmax(dim=-1).type(
                cls_labels.dtype) == cls_labels).sum())
            bbox_mae = float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
            metric.add(cls_acc, cls_labels.numel(),
                      bbox_mae, bbox_labels.numel())
            timer.stop()
        
        # 计算训练集指标
        train_cls_err = 1 - metric[0] / metric[1]
        train_bbox_mae = metric[2] / metric[3]
        
        # 计算验证集指标
        if val_iter is not None:
            val_cls_err, val_bbox_mae = evaluate_accuracy(net, val_iter, device)
        else:
            val_cls_err, val_bbox_mae = None, None
        
        # 更新动画
        animator.add(epoch + 1, 
                    (train_cls_err, train_bbox_mae,
                     val_cls_err, val_bbox_mae))
    
    # 设置并保存最终结果
    final_results = (f'Final Results:\n'
                    f'Train Class Error: {train_cls_err:.3f}\n'
                    f'Train Bbox MAE: {train_bbox_mae:.3f}\n'
                    f'Val Class Error: {val_cls_err:.3f}\n'
                    f'Val Bbox MAE: {val_bbox_mae:.3f}\n')
    animator.save()
    
    # 打印训练统计信息
    print(f'{title} with learning rate {lr}:')
    print(final_results)
    print(f'{len(train_iter.dataset) / timer.sum():.1f} examples/sec on {device}')
    
    # 在训练循环结束后，保存模型
    if save_model:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_save_path = f'checkpoints/ssd_model_{timestamp}.pth'
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': num_epochs,
            'train_cls_err': train_cls_err,
            'train_bbox_mae': train_bbox_mae,
            'val_cls_err': val_cls_err,
            'val_bbox_mae': val_bbox_mae
        }, model_save_path)
        print(f'模型已保存至 {model_save_path}')
