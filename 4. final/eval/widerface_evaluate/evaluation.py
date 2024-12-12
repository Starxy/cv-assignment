"""
WiderFace评估代码
作者: wondervictor
邮箱: tianhengcheng@gmail.com
版权所有@wondervictor
"""

import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
from bbox import bbox_overlaps


def get_gt_boxes(gt_dir):
    """获取真实标注框
    参数:
        gt_dir: 包含wider_face_val.mat、wider_easy_val.mat、wider_medium_val.mat、wider_hard_val.mat的目录
    返回:
        facebox_list: 人脸框列表
        event_list: 事件列表
        file_list: 文件列表
        hard_gt_list: 困难样本列表
        medium_gt_list: 中等样本列表
        easy_gt_list: 简单样本列表
    """

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list

def read_pred_file(filepath):
    """读取预测结果文件
    参数:
        filepath: 预测结果文件路径
    返回:
        img_file: 图片文件名
        boxes: 预测框数组
    """

    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')], lines))).astype('float')
    return img_file.split('/')[-1], boxes


def get_preds(pred_dir):
    """获取所有预测结果
    参数:
        pred_dir: 预测结果目录
    返回:
        boxes: 预测框字典,格式为{event:{img:boxes}}
    """
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm.tqdm(events)

    for event in pbar:
        pbar.set_description('读取预测结果')
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname.rstrip('.jpg')] = _boxes
        boxes[event] = current_event
    return boxes


def norm_score(pred):
    """归一化预测分数
    参数:
        pred: 预测结果字典,格式为{key: [[x1,y1,x2,y2,s]]}
    """

    max_score = 0
    min_score = 1

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score)/diff


def image_eval(pred, gt, ignore, iou_thresh):
    """单张图片评估
    参数:
        pred: Nx5的预测框数组
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

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

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
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
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

def evaluation(pred, gt_path, iou_thresh=0.5):
    """评估函数
    参数:
        pred: 预测结果目录,目录结构为:
             pred/
                event_name1/
                    image_name1.txt  # 每个txt文件包含检测框信息
                    image_name2.txt
                    ...
                event_name2/
                    image_name1.txt
                    image_name2.txt
                    ...
                ...
        gt_path: 真实标注目录
        iou_thresh: IOU阈值,默认0.5
    """
    pred = get_preds(pred)
    norm_score(pred)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)
    thresh_num = 1000
    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    aps = []
    for setting_id in range(3):
        # 不同难度设置
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')
        # [hard, medium, easy]
        pbar = tqdm.tqdm(range(event_num))
        for i in pbar:
            pbar.set_description('处理 {}'.format(settings[setting_id]))
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            pred_list = pred[event_name]
            sub_gt_list = gt_list[i][0]
            # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
            gt_bbx_list = facebox_list[i][0]

            for j in range(len(img_list)):
                pred_info = pred_list[str(img_list[j][0][0])]

                gt_boxes = gt_bbx_list[j][0].astype('float')
                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index-1] = 1
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)

                _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

                pr_curve += _img_pr_info
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = voc_ap(recall, propose)
        aps.append(ap)

    print("==================== Results ====================")
    print("Easy   Val AP: {}".format(aps[0]))
    print("Medium Val AP: {}".format(aps[1]))
    print("Hard   Val AP: {}".format(aps[2]))
    print("=================================================")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred')
    parser.add_argument('-g', '--gt', default='C:/Project/ai/cv-assignment/4. final/eval/widerface_evaluate/ground_truth')

    args = parser.parse_args()
    evaluation(args.pred, args.gt)












