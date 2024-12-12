"""
WiderFace评估代码
作者: wondervictor
邮箱: tianhengcheng@gmail.com
版权所有@wondervictor

"""

import os
import tqdm
import numpy as np
from bbox import bbox_overlaps

def read_gt_files(gt_file):
    """读取标注数据
    
    参数:
        gt_file: 标注文件路径
        
    返回:
        gt_dict: {image_path: boxes} 
        boxes格式为 Nx4 数组 [x,y,w,h]
    """
    gt_dict = {}
    
    with open(gt_file, 'r') as f:
        lines = f.readlines()
        
    idx = 0
    while idx < len(lines):
        # 读取图片路径
        img_path = lines[idx].strip()
        # 读取框数量
        box_num = int(lines[idx + 1])
        # 读取框坐标
        boxes = []
        for i in range(box_num):
            box = list(map(float, lines[idx + 2 + i].strip().split()))
            boxes.append(box)
        gt_dict[img_path] = np.array(boxes)
        idx += box_num + 2
        
    return gt_dict

def read_pred_file(filepath):
    """读取单个预测结果文件"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_path = lines[0].strip()
        box_num = int(lines[1])
        boxes = []
        for i in range(box_num):
            box = list(map(float, lines[i + 2].strip().split()))
            boxes.append(box)
    return img_path, np.array(boxes)

def get_preds(pred_dir):
    """读取所有预测结果
    
    返回:
        pred_dict: {image_path: boxes}
        boxes格式为 Nx5 数组 [x,y,w,h,score]
    """
    pred_dict = {}
    
    # 递归遍历目录下所有txt文件
    for root, _, files in os.walk(pred_dir):
        for f in files:
            if not f.endswith('.txt') or f == 'process.txt':
                continue
            filepath = os.path.join(root, f)
            img_path, boxes = read_pred_file(filepath)
            img_path = img_path + '.jpg'
            pred_dict[img_path] = boxes
            
    return pred_dict

def image_eval(pred, gt, ignore, iou_thresh):
    """单张图片评估
    参数:
        pred: Nx5的预测框数组,每行为[x1,y1,x2,y2,score]
        gt: Nx4的真实框数组
        ignore: 忽略标记数组
        iou_thresh: IOU阈值
    返回:
        pred_recall: 预测框召回率
        proposal_list: 提议框列表
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    # 将预测框和真值框转换为x2,y2格式
    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    # 计算IOU
    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    # 按置信度从高到低排序
    conf_sort_idx = np.argsort(_pred[:, 4])[::-1]
    _pred = _pred[conf_sort_idx]
    overlaps = overlaps[conf_sort_idx]
    proposal_list = proposal_list[conf_sort_idx]
    pred_recall = pred_recall[conf_sort_idx]

    # 遍历每个预测框
    for h in range(_pred.shape[0]):
        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        
        # 如果IOU大于阈值
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:  # 忽略的真值框
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:  # 未匹配的真值框
                recall_list[max_idx] = 1

        # 统计召回的真值框数量
        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
        
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    """计算单张图片的PR信息
    参数:
        thresh_num: 阈值数量
        pred_info: 预测信息
        proposal_list: 提议框列表
        pred_recall: 预测框召回率
    返回:
        pr_info: PR信息数组
    """
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t+1)/thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index+1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    """计算数据集的PR信息
    参数:
        thresh_num: 阈值数量
        pr_curve: PR曲线
        count_face: 人脸计数
    返回:
        _pr_curve: 处理后的PR曲线
    """
    _pr_curve = np.zeros((thresh_num, 2))
    
    # 处理分母为零的情况
    for i in range(thresh_num):
        # 精确率计算：预测正确的数量/预测总数
        _pr_curve[i, 0] = float(pr_curve[i, 1]) / float(pr_curve[i, 0]) if pr_curve[i, 0] > 0 else 0
        
        # 召回率计算：预测正确的数量/真实标注总数
        _pr_curve[i, 1] = float(pr_curve[i, 1]) / float(count_face) if count_face > 0 else 0
        
    return _pr_curve


def voc_ap(rec, prec):
    """计算平均精度(AP)
    参数:
        rec: 召回率数组
        prec: 精确率数组
    返回:
        ap: 平均精度
    """

    # 正确的AP计算
    # 首先在末尾添加哨兵值
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # 计算精度包络
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 寻找X轴(召回率)变化的点
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # 计算AP = Σ(Δrecall × precision)
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate(pred_dir, gt_file, iou_thresh=0.5):
    """评估函数
    
    参数:
        pred_dir: 预测结果目录
        gt_file: 标注文件路径
        iou_thresh: IOU阈值
        
    返回:
        ap: 平均精度
    """
    # 读取数据
    pred_dict = get_preds(pred_dir)
    gt_dict = read_gt_files(gt_file)
    
    # 归一化预测分数
    max_score = 0
    min_score = 1
    for boxes in pred_dict.values():
        if len(boxes) == 0:
            continue
        max_score = max(max_score, np.max(boxes[:, -1]))
        min_score = min(min_score, np.min(boxes[:, -1]))
    score_range = max_score - min_score
    
    for boxes in pred_dict.values():
        if len(boxes) > 0:
            boxes[:, -1] = (boxes[:, -1] - min_score) / score_range
    
    # 评估参数
    thresh_num = 1000
    count_face = 0
    pr_curve = np.zeros((thresh_num, 2)).astype('float')
    
    # 评估每张图片
    pbar = tqdm.tqdm(gt_dict.items())
    for img_path, gt_boxes in pbar:
        pbar.set_description('评估中')
        
        # 跳过没有预测结果的图片
        if img_path not in pred_dict:
            continue
            
        pred_boxes = pred_dict[img_path]
        count_face += len(gt_boxes)
        
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue
            
        # 评估单张图片
        ignore = np.ones(len(gt_boxes))  # 所有gt框都参与评估
        pred_recall, proposal_list = image_eval(pred_boxes, gt_boxes, ignore, iou_thresh)
        _img_pr_info = img_pr_info(thresh_num, pred_boxes, proposal_list, pred_recall)
        pr_curve += _img_pr_info
    
    # 计算PR和AP
    pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)
    propose = pr_curve[:, 0]
    recall = pr_curve[:, 1]
    ap = voc_ap(recall, propose)
    
    return ap

if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', help='预测结果目录')
    parser.add_argument('-g', '--gt', help='标注文件路径')
    args = parser.parse_args()
    
    # 遍历预测结果目录下的每个子目录
    for model_name in os.listdir(args.pred):
        model_pred_dir = os.path.join(args.pred, model_name)
        if os.path.isdir(model_pred_dir):
            print(f"\n评估模型: {model_name}")
            ap = evaluate(model_pred_dir, args.gt)
            print(f"{model_name} mAP: {ap}")
