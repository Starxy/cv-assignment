"""
SSD目标检测模型训练模块

@Author: DONG Jixing
@Date: 2024-11-24
"""

import torch
from torch import nn
from torchvision.ops import box_iou  # 替换自定义的box_iou
from ssd import TinySSD
from utils import Timer, Animator, Accumulator

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
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
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
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    """计算分类和边界框预测的损失"""
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss = nn.L1Loss(reduction='none')
    
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

def calculate_map(pred_boxes, pred_scores, pred_labels, true_boxes, true_labels, iou_threshold=0.5):
    """计算mAP
    Args:
        pred_boxes: 预测框 [N, 4]
        pred_scores: 预测分数 [N]
        pred_labels: 预测类别 [N]
        true_boxes: 真实框 [M, 4]
        true_labels: 真实类别 [M]
        iou_threshold: IoU阈值
    """
    ap_per_class = {}
    
    # 对每个类别分别计算AP
    unique_labels = torch.unique(torch.cat([pred_labels, true_labels]))
    
    for c in unique_labels:
        # 获取该类别的预测和真实框
        class_pred_mask = pred_labels == c
        class_true_mask = true_labels == c
        
        if not class_true_mask.any():  # 如果没有真实框，跳过
            continue
            
        class_pred_boxes = pred_boxes[class_pred_mask]
        class_pred_scores = pred_scores[class_pred_mask]
        class_true_boxes = true_boxes[class_true_mask]
        
        if len(class_pred_boxes) == 0:  # 如果没有预测框，AP为0
            ap_per_class[int(c)] = 0
            continue
            
        # 按置信度排序
        sorted_indices = torch.argsort(class_pred_scores, descending=True)
        class_pred_boxes = class_pred_boxes[sorted_indices]
        class_pred_scores = class_pred_scores[sorted_indices]
        
        # 计算TP和FP
        tp = torch.zeros(len(class_pred_boxes))
        fp = torch.zeros(len(class_pred_boxes))
        
        for pred_idx, pred_box in enumerate(class_pred_boxes):
            if len(class_true_boxes) == 0:
                fp[pred_idx] = 1
                continue
                
            # 计算与所有真实框的IoU
            ious = box_iou(pred_box.unsqueeze(0), class_true_boxes)
            max_iou, max_idx = torch.max(ious, dim=1)
            
            if max_iou >= iou_threshold:
                tp[pred_idx] = 1
                # 移除已匹配的真实框
                class_true_boxes = torch.cat([class_true_boxes[:max_idx], 
                                           class_true_boxes[max_idx+1:]])
            else:
                fp[pred_idx] = 1
        
        # 计算precision和recall
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        recalls = tp_cumsum / len(class_true_mask)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # 计算AP
        ap = torch.trapz(precisions, recalls)
        ap_per_class[int(c)] = float(ap)
    
    # 计算mAP
    return sum(ap_per_class.values()) / len(ap_per_class) if ap_per_class else 0

def evaluate_accuracy(net, data_iter, device):
    """评估模型的分类和边界框预测准确性，包括mAP"""
    net.eval()
    metric = Accumulator(4)  # 分类正确数、分类总数、bbox误差和、bbox总数
    all_maps = []
    
    with torch.no_grad():
        for features, target in data_iter:
            X, Y = features.to(device), target.to(device)
            # 获取预测结果
            anchors, cls_preds, bbox_preds = net(X)
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            
            # 获取预测的边界框、分数和类别
            pred_scores = torch.max(cls_preds.softmax(dim=-1), dim=-1)[0]
            pred_labels = cls_preds.argmax(dim=-1)
            
            # 对每个批次计算mAP
            batch_map = calculate_map(
                bbox_preds[bbox_masks.bool()].reshape(-1, 4),
                pred_scores[bbox_masks.bool().squeeze(-1)],
                pred_labels[bbox_masks.bool().squeeze(-1)],
                Y[:, :, 1:],  # 真实框坐标
                Y[:, :, 0].long()  # 真实框类别
            )
            all_maps.append(batch_map)
            
            # 原有的评估指标
            cls_acc = float((cls_preds.argmax(dim=-1).type(
                cls_labels.dtype) == cls_labels).sum())
            bbox_mae = float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
            
            metric.add(cls_acc, cls_labels.numel(),
                      bbox_mae, bbox_labels.numel())
    
    cls_err = 1 - metric[0] / metric[1]
    bbox_mae = metric[2] / metric[3]
    mean_map = sum(all_maps) / len(all_maps)
    
    return cls_err, bbox_mae, mean_map

def train(net, train_iter, val_iter, num_epochs, lr, device, title="SSD Training"):
    """训练SSD模型的主函数"""
    print(f'training on {device}')
    net.to(device)
    
    # 定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=5e-4)
    
    # 创建动画器
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                       legend=['train cls error', 'train bbox mae', 'train mAP',
                              'val cls error', 'val bbox mae', 'val mAP'],
                       title=title)
    
    timer = Timer()
    for epoch in range(num_epochs):
        # 训练阶段
        metric = Accumulator(4)
        net.train()
        all_train_maps = []  # 用于存储训练集的mAP
        
        for features, target in train_iter:
            timer.start()
            optimizer.zero_grad()
            X, Y = features.to(device), target.to(device)
            
            # 前向传播
            anchors, cls_preds, bbox_preds = net(X)
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            
            # 计算损失并反向传播
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.mean().backward()
            optimizer.step()
            
            # 计算训练指标
            pred_scores = torch.max(cls_preds.softmax(dim=-1), dim=-1)[0]
            pred_labels = cls_preds.argmax(dim=-1)
            
            # 计算当前批次的mAP
            batch_map = calculate_map(
                bbox_preds[bbox_masks.bool()].reshape(-1, 4),
                pred_scores[bbox_masks.bool().squeeze(-1)],
                pred_labels[bbox_masks.bool().squeeze(-1)],
                Y[:, :, 1:],
                Y[:, :, 0].long()
            )
            all_train_maps.append(batch_map)
            
            # 评估其他指标
            cls_acc = float((cls_preds.argmax(dim=-1).type(
                cls_labels.dtype) == cls_labels).sum())
            bbox_mae = float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
            metric.add(cls_acc, cls_labels.numel(),
                      bbox_mae, bbox_labels.numel())
            timer.stop()
        
        # 计算训练集指标
        train_cls_err = 1 - metric[0] / metric[1]
        train_bbox_mae = metric[2] / metric[3]
        train_map = sum(all_train_maps) / len(all_train_maps)  # 计算平均mAP
        
        # 计算验证集指标
        if val_iter is not None:
            val_cls_err, val_bbox_mae, val_map = evaluate_accuracy(net, val_iter, device)
        else:
            val_cls_err, val_bbox_mae, val_map = None, None, None
        
        # 更新动画
        animator.add(epoch + 1, 
                    (train_cls_err, train_bbox_mae, train_map,
                     val_cls_err, val_bbox_mae, val_map))
    
    # 设置并保存最终结果
    final_results = (f'Final Results:\n'
                    f'Train Class Error: {train_cls_err:.3f}\n'
                    f'Train Bbox MAE: {train_bbox_mae:.3f}\n'
                    f'Train mAP: {train_map:.3f}\n'
                    f'Val Class Error: {val_cls_err:.3f}\n'
                    f'Val Bbox MAE: {val_bbox_mae:.3f}\n'
                    f'Val mAP: {val_map:.3f}')
    # animator.set_results(final_results)
    animator.save()
    
    # 打印训练统计信息
    print(f'{title} with learning rate {lr}:')
    print(final_results)
    print(f'{len(train_iter.dataset) / timer.sum():.1f} examples/sec on {device}')
