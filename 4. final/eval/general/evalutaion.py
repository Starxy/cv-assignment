"""
通用目标检测评估代码
"""

import os
import tqdm
import pickle
import numpy as np
from bbox import bbox_overlaps

def get_gt_boxes_from_txt(gt_path, cache_dir):
    """从txt文件加载ground truth标注框
    
    Args:
        gt_path: ground truth标注文件路径,格式为:
                image_name.jpg
                num_boxes
                x1 y1 x2 y2 class_id
                x1 y1 x2 y2 class_id
                ...
        cache_dir: 缓存目录
    Returns:
        boxes: 标注框字典 {image_name: array([[x1,y1,x2,y2,class_id],...]}
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            boxes = pickle.load(f)
        return boxes

    boxes = {}
    with open(gt_path, 'r') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        img_name = lines[i].strip()
        num_boxes = int(lines[i+1])
        boxes[img_name] = []
        
        for b in range(num_boxes):
            box = [float(x) for x in lines[i+2+b].split()]
            boxes[img_name].append(box)
            
        boxes[img_name] = np.array(boxes[img_name])
        i += num_boxes + 2
    
    with open(cache_file, 'wb') as f:
        pickle.dump(boxes, f)
        
    return boxes

def read_pred_file(filepath):
    """读取预测结果文件
    
    Args:
        filepath: 预测文件路径,格式为:
                x1 y1 x2 y2 score class_id
                x1 y1 x2 y2 score class_id 
                ...
    Returns:
        boxes: 预测框数组 Nx6 (x1,y1,x2,y2,score,class_id)
    """
    boxes = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            box = [float(x) for x in line.strip().split()]
            boxes.append(box)
    return np.array(boxes)

def get_preds(pred_dir):
    """读取所有预测结果
    
    Args:
        pred_dir: 预测结果目录,包含每张图片的txt预测文件
    Returns:
        boxes: 预测框字典 {image_name: array([[x1,y1,x2,y2,score,class_id],...]}
    """
    boxes = {}
    for f in tqdm.tqdm(os.listdir(pred_dir), desc='Reading Predictions'):
        if not f.endswith('.txt'):
            continue
        img_name = f.replace('.txt','')
        boxes[img_name] = read_pred_file(os.path.join(pred_dir, f))
    return boxes

def image_eval(pred, gt, iou_thresh):
    """单张图片评估
    
    Args:
        pred: 预测框 Nx6 (x1,y1,x2,y2,score,class_id)
        gt: ground truth框 Mx5 (x1,y1,x2,y2,class_id)
        iou_thresh: IOU阈值
    Returns:
        tp: 真阳性数组
        fp: 假阳性数组
    """
    num_pred = pred.shape[0]
    tp = np.zeros(num_pred)
    fp = np.zeros(num_pred)
    
    if gt.shape[0] == 0:
        fp[:] = 1
        return tp, fp
        
    # 计算IoU
    overlaps = bbox_overlaps(pred[:,:4], gt[:,:4])
    
    # 对每个预测框
    for i in range(num_pred):
        if pred[i,-1] != gt[0,-1]:  # 类别不匹配
            fp[i] = 1
            continue
            
        max_overlap = overlaps[i].max()
        max_idx = overlaps[i].argmax()
        
        if max_overlap >= iou_thresh:
            if gt[max_idx,-1] == pred[i,-1]:  # 类别匹配
                tp[i] = 1
                overlaps[:,max_idx] = -1  # 标记该gt已匹配
            else:
                fp[i] = 1
        else:
            fp[i] = 1
            
    return tp, fp

def calculate_ap_per_class(pred_boxes, gt_boxes, class_id, iou_thresh):
    """计算每个类别的AP值
    
    Args:
        pred_boxes: 预测框字典
        gt_boxes: ground truth框字典
        class_id: 类别ID
        iou_thresh: IOU阈值
    Returns:
        ap: 平均精度
    """
    tp_all = []
    fp_all = []
    scores_all = []
    
    total_gt = 0
    
    # 对每张图片
    for img_name in pred_boxes.keys():
        pred = pred_boxes[img_name]
        gt = gt_boxes.get(img_name, np.zeros((0,5)))
        
        # 筛选当前类别
        pred_cls = pred[pred[:,-1] == class_id]
        gt_cls = gt[gt[:,-1] == class_id]
        
        total_gt += len(gt_cls)
        
        if len(pred_cls) == 0:
            continue
            
        # 按置信度排序
        sort_idx = np.argsort(-pred_cls[:,4])
        pred_cls = pred_cls[sort_idx]
        
        tp, fp = image_eval(pred_cls, gt_cls, iou_thresh)
        
        tp_all.append(tp)
        fp_all.append(fp)
        scores_all.append(pred_cls[:,4])
        
    if not tp_all:
        return 0.0
        
    tp_all = np.concatenate(tp_all)
    fp_all = np.concatenate(fp_all)
    scores_all = np.concatenate(scores_all)
    
    # 按置信度排序
    sort_idx = np.argsort(-scores_all)
    tp_all = tp_all[sort_idx]
    fp_all = fp_all[sort_idx]
    
    # 计算累积值
    tp_cum = np.cumsum(tp_all)
    fp_cum = np.cumsum(fp_all)
    
    # 计算查准率和查全率
    prec = tp_cum / (tp_cum + fp_cum)
    rec = tp_cum / total_gt if total_gt > 0 else np.zeros_like(tp_cum)
    
    # 计算AP
    ap = calculate_ap(rec, prec)
    
    return ap

def calculate_ap(recall, precision):
    """计算AP值(使用VOC2007的11点插值方法)
    
    Args:
        recall: 召回率数组
        precision: 精确率数组
    Returns:
        ap: 平均精度
    """
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.
    return ap

def evaluate(pred_dir, gt_path, iou_thresh=0.5, cache_dir='./cache'):
    """评估检测器性能
    
    Args:
        pred_dir: 预测结果目录
        gt_path: ground truth标注文件路径
        iou_thresh: IOU阈值,默认0.5
        cache_dir: 缓存目录
    Returns:
        mAP: 平均精度均值
        ap_dict: 每个类别的AP字典
    """
    # 加载数据
    gt_boxes = get_gt_boxes_from_txt(gt_path, cache_dir)
    pred_boxes = get_preds(pred_dir)
    
    # 获取所有类别
    classes = set()
    for boxes in gt_boxes.values():
        classes.update(boxes[:,-1].astype(int))
    classes = sorted(list(classes))
    
    # 计算每个类别的AP
    ap_dict = {}
    for cls in tqdm.tqdm(classes, desc='Calculating AP'):
        ap = calculate_ap_per_class(pred_boxes, gt_boxes, cls, iou_thresh)
        ap_dict[cls] = ap
        
    # 计算mAP
    mAP = np.mean(list(ap_dict.values()))
    
    # 打印结果
    print("\n==================== Results ====================")
    for cls, ap in ap_dict.items():
        print(f"Class {cls} AP: {ap:.4f}")
    print(f"mAP: {mAP:.4f}")
    print("===============================================")
    
    return mAP, ap_dict

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True, help='预测结果目录')
    parser.add_argument('--gt_path', required=True, help='ground truth文件路径')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IOU阈值')
    parser.add_argument('--cache_dir', default='./cache', help='缓存目录')
    args = parser.parse_args()
    
    evaluate(args.pred_dir, args.gt_path, args.iou_thresh, args.cache_dir)