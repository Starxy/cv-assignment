"""
WiderFace评估代码
作者: wondervictor
邮箱: tianhengcheng@gmail.com
版权所有@wondervictor

# 数据准备阶段

## 1. 读取预测结果
get_preds(pred_dir)
    ├── 遍历所有event目录
    └── read_pred_file() # 读取每个txt中的预测框

## 2. 分数归一化
norm_score(pred)
    └── 将所有预测分数归一化到[0,1]区间

## 3. 读取真值数据
get_gt_boxes(gt_dir)
    └── 加载各个难度级别的ground truth数据

# 评估循环阶段
## 对每个难度级别(easy/medium/hard)进行评估
for setting_id in range(3):
    └── 对每个event循环:
        └── 对每张图片进行评估:
            ├── image_eval() # 计算单张图片的召回率
            │   ├── 计算预测框和真值框的IOU
            │   └── 返回pred_recall和proposal_list
            │
            ├── img_pr_info() # 计算PR信息
            │   └── 在不同阈值下计算precision和recall
            │
            └── 累加PR曲线信息

    ├── dataset_pr_info() # 计算整个数据集的PR信息
    └── voc_ap() # 计算AP值

# 核心评估逻辑
image_eval():
    1. 将预测框和真值框转换为[x1,y1,x2,y2]格式
    2. 计算所有预测框和真值框之间的IOU
    3. 对每个预测框:
        - 找到最大IOU对应的真值框
        - 根据IOU阈值和ignore标记更新召回列表
    4. 返回预测召回率和提议列表

img_pr_info():
    1. 在不同置信度阈值下:
        - 统计有效预测框数量
        - 计算对应的precision和recall

dataset_pr_info():
    1. 计算整个数据集的precision: TP/(TP+FP)
    2. 计算整个数据集的recall: TP/total_faces

voc_ap():
    1. 使用11点插值法计算AP
    2. 计算PR曲线下的面积
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
    
    该函数用于读取WIDER FACE数据集的验证集标注信息,包括:
    1. wider_face_val.mat: 包含所有人脸框标注、事件列表和文件列表
    2. wider_hard_val.mat: 困难样本的标注
    3. wider_medium_val.mat: 中等难度样本的标注  
    4. wider_easy_val.mat: 简单样本的标注
    
    具体实现:
    1. 使用loadmat()函数分别加载4个.mat文件
    2. 从wider_face_val.mat中提取:
       - face_bbx_list: 人脸框坐标列表
       - event_list: 场景事件类别列表
       - file_list: 图片文件名列表
    3. 从其他3个.mat文件中提取对应难度的ground truth列表
    
    参数:
        gt_dir: 包含wider_face_val.mat、wider_easy_val.mat、wider_medium_val.mat、wider_hard_val.mat的目录
        
    返回:
        facebox_list: 人脸框列表,包含所有图片中人脸的坐标信息
        event_list: 事件列表,标识每张图片属于哪个场景事件
        file_list: 文件列表,包含所有图片的文件名
        hard_gt_list: 困难样本的ground truth列表
        medium_gt_list: 中等难度样本的ground truth列表  
        easy_gt_list: 简单样本的ground truth列表
    """

    # 加载.mat格式的标注文件
    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))
    
    front_camera_mat = loadmat(os.path.join(gt_dir, 'wider_front_camera_val.mat'))
    # 从wider_face_val.mat中提取基本信息
    facebox_list = gt_mat['face_bbx_list']  # 人脸框坐标
    event_list = gt_mat['event_list']       # 事件类别
    file_list = gt_mat['file_list']         # 文件名
    
    # 提取不同难度的ground truth
    hard_gt_list = hard_mat['gt_list']      # 困难样本
    medium_gt_list = medium_mat['gt_list']  # 中等样本
    easy_gt_list = easy_mat['gt_list']      # 简单样本

    front_camera_gt_list = front_camera_mat['gt_list'] # 前置相机样本

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list, front_camera_gt_list

def read_pred_file(filepath):
    """读取单个图像预测结果文件
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
    
    该函数用于读取预测结果目录下的所有预测框信息,并组织成字典格式返回。
    
    具体实现:
    1. 遍历预测结果目录下的所有事件文件夹
    2. 对每个事件文件夹:
       - 读取其中所有图片的预测结果文件(.txt)
       - 将每个图片的预测框信息存入字典
    3. 最终返回嵌套字典结构:{事件名:{图片名:预测框数组}}
    
    参数:
        pred_dir: 预测结果目录,包含多个事件子目录
        
    返回:
        boxes: 预测框字典,格式为{event:{img:boxes}}
              - event: 事件名
              - img: 图片名(不含扩展名)
              - boxes: 该图片的预测框数组
    """
    # 获取所有事件文件夹
    events = [f for f in os.listdir(pred_dir) if os.path.isdir(os.path.join(pred_dir, f))]
    boxes = dict()
    # 使用进度条显示处理进度
    pbar = tqdm.tqdm(events)

    for event in pbar:
        pbar.set_description('读取预测结果')
        # 获取当前事件目录
        event_dir = os.path.join(pred_dir, event)
        # 获取该事件下所有图片的预测结果文件
        event_images = os.listdir(event_dir)
        current_event = dict()
        # 处理每个预测结果文件
        for imgtxt in event_images:
            # 读取预测结果文件,获取图片名和预测框
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            # 存入字典,移除.jpg后缀
            current_event[imgname.rstrip('.jpg')] = _boxes
        boxes[event] = current_event
    return boxes


def norm_score(pred):
    """归一化预测分数
    
    该函数用于对预测框的置信度分数进行归一化处理,将所有分数映射到[0,1]区间。
    
    具体实现:
    1. 首先遍历所有预测框,找出最大和最小置信度分数
    2. 然后使用min-max归一化公式:
       normalized_score = (score - min_score)/(max_score - min_score)
    3. 将所有预测框的置信度更新为归一化后的值
    
    参数:
        pred: 预测结果字典,格式为{event_name: {image_name: boxes}}
             其中boxes为Nx5数组,每行为[x1,y1,x2,y2,score]
    """

    # 初始化最大最小分数
    max_score = 0  # 记录所有预测框中的最大置信度
    min_score = 1  # 记录所有预测框中的最小置信度

    # 第一次遍历,找出全局最大最小分数
    for _, k in pred.items():  # 遍历每个事件
        for _, v in k.items():  # 遍历每张图片的预测框
            if len(v) == 0:  # 跳过空预测
                continue
            _min = np.min(v[:, -1])  # 当前图片预测框的最小分数
            _max = np.max(v[:, -1])  # 当前图片预测框的最大分数
            max_score = max(_max, max_score)  # 更新全局最大分数
            min_score = min(_min, min_score)  # 更新全局最小分数

    # 计算分数范围
    diff = max_score - min_score
    
    # 第二次遍历,执行归一化
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            # 对每个预测框的置信度进行归一化
            v[:, -1] = (v[:, -1] - min_score)/diff

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
    for i in range(thresh_num):
        if pr_curve[i, 0] != 0:  # 确保分母不为零
            _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        else:
            _pr_curve[i, 0] = 0  # 或者设置为其他默认值
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
    
    该函数用于评估人脸检测模型在WIDER FACE数据集上的性能。
    
    具体实现:
    1. 数据准备:
       - 读取预测结果并归一化分数
       - 加载真实标注数据
    2. 对每个难度级别(easy/medium/hard)进行评估:
       - 遍历每个事件(event)
       - 对每张图片:
         * 计算预测框和真实框的匹配情况
         * 统计PR曲线信息
       - 计算整个数据集的PR曲线
       - 计算AP值
    3. 输出三个难度级别的AP结果
    
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
    # 1. 数据准备
    pred = get_preds(pred)  # 读取预测结果
    norm_score(pred)        # 归一化预测分数
    # 加载真实标注数据
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list, front_camera_gt_list = get_gt_boxes(gt_path)
    
    event_num = len(event_list)      # 事件数量
    thresh_num = 1000                # 阈值数量
    settings = ['easy', 'medium', 'hard', 'front_camera']  # 难度设置
    # 存储不同难度级别(easy/medium/hard)的ground truth列表
    # setting_gts中存储的是不同难度级别下每个图片中需要评估的真值框的索引信息
    # 而不是具体的边框坐标。具体的边框坐标存储在facebox_list中
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list, front_camera_gt_list]  
    aps = []  # 存储不同难度的AP值
    
    # 对每个难度级别(easy/medium/hard/front_camera)进行评估
    for setting_id in range(4):
        # 获取当前难度级别的ground truth列表(索引信息)
        gt_list = setting_gts[setting_id]  
        count_face = 0  # 统计人脸总数
        pr_curve = np.zeros((thresh_num, 2)).astype('float')  # 存储PR曲线点
        
        # 使用进度条遍历每个事件场景
        pbar = tqdm.tqdm(range(event_num))
        for i in pbar:  
            pbar.set_description('处理 {}'.format(settings[setting_id]))
            
            # 获取当前事件的信息
            event_name = str(event_list[i][0][0])  # 事件名称
            img_list = file_list[i][0]  # 该事件下的所有图片
            pred_list = pred[event_name]  # 该事件的所有预测结果
            sub_gt_list = gt_list[i][0]  # 该事件的ground truth标记(索引信息)
            gt_bbx_list = facebox_list[i][0]  # 该事件的所有真值框坐标

            # 对当前事件下的每张图片进行评估
            for j in range(len(img_list)):
                # 获取当前图片的预测框和真值框
                img_name = str(img_list[j][0][0])
                pred_info = pred_list[img_name]  # 预测框列表
                gt_boxes = gt_bbx_list[j][0].astype('float')  # 真值框列表(坐标信息)
                
                # 获取当前难度级别下需要评估的真值框索引
                keep_index = sub_gt_list[j][0]  
                count_face += len(keep_index)  # 累加有效人脸数

                # 如果没有真值框或预测框,跳过评估
                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                    
                # 标记哪些真值框需要参与评估(1表示评估,0表示忽略)
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index-1] = 1
                    
                # 评估单张图片:
                # 1. image_eval计算每个预测框是否匹配到真值框
                # 2. img_pr_info统计不同置信度阈值下的precision和recall
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)
                _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
                pr_curve += _img_pr_info  # 累加PR统计信息

        # 计算数据集级别的PR曲线
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)
        propose = pr_curve[:, 0]  # 精确率
        recall = pr_curve[:, 1]   # 召回率
        
        # 计算AP
        ap = voc_ap(recall, propose)
        aps.append(ap)

    # 3. 输出结果
    print("==================== Results ====================")
    print("Easy         Val AP: {}".format(aps[0]))
    print("Medium       Val AP: {}".format(aps[1]))
    print("Hard         Val AP: {}".format(aps[2]))
    print("Front Camera Val AP: {}".format(aps[3]))
    print("=================================================")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred')
    parser.add_argument('-g', '--gt', default='C:/Project/ai/cv-assignment/4. final/eval/widerface_evaluate/ground_truth')

    args = parser.parse_args()
    evaluation(args.pred, args.gt)












