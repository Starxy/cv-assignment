import os
import json
import numpy as np
from scipy.io import loadmat

def inspect_mat(gt_dir):
    """检查mat文件的内容结构"""
    
    # 加载主标注文件
    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    
    # 获取第一个事件的信息作为示例
    event_name = str(gt_mat['event_list'][0][0][0])
    print(f"\n事件名称示例: {event_name}")
    
    # 获取该事件下第一张图片的信息
    img_name = str(gt_mat['file_list'][0][0][0][0])
    print(f"图片名称示例: {img_name}")
    
    # 获取该图片的人脸框
    face_boxes = gt_mat['face_bbx_list'][0][0][0][0]
    print(f"人脸框示例(x,y,w,h格式):\n{face_boxes}")
    
    # 加载难度级别文件
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    
    # 显示第一个事件的难度级别图片索引
    print("\n第一个事件中不同难度级别的图片索引:")
    print(f"简单: {easy_mat['gt_list'][0][0]}")
    print(f"中等: {medium_mat['gt_list'][0][0]}")
    print(f"困难: {hard_mat['gt_list'][0][0]}")
    
    # 显示一些基本统计信息
    print(f"\n总事件数: {len(gt_mat['event_list'])}")
    print(f"第一个事件的图片数: {len(gt_mat['file_list'][0][0])}")


def mat2json(gt_dir, output_dir):
    """将WIDER FACE的mat文件转换为JSON格式
    
    参数:
        gt_dir: mat文件所在目录
        output_dir: 输出JSON文件目录
    """
    # 加载mat文件
    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))
    
    # 构建数据结构
    dataset = {
        'images': [],
        'annotations': {}
    }
    
    # 处理事件和文件列表
    for event_idx, event in enumerate(gt_mat['event_list']):
        event_name = str(event[0][0])
        
        # 处理每个事件下的图片
        for img_idx, img_name in enumerate(gt_mat['file_list'][event_idx][0]):
            img_name = str(img_name[0][0])
            
            # 获取标注框
            face_boxes = gt_mat['face_bbx_list'][event_idx][0][img_idx][0]
            if face_boxes.size > 0:
                face_boxes = face_boxes.tolist()
            else:
                face_boxes = []
                
            # 获取难度标记
            is_easy = 1 if img_idx in easy_mat['gt_list'][event_idx][0] else 0
            is_medium = 1 if img_idx in medium_mat['gt_list'][event_idx][0] else 0
            is_hard = 1 if img_idx in hard_mat['gt_list'][event_idx][0] else 0
            
            # 构建图片信息
            img_info = {
                'file_name': f"{event_name}/{img_name}",
                'face_boxes': face_boxes,
                'difficulty': {
                    'easy': is_easy,
                    'medium': is_medium,
                    'hard': is_hard
                }
            }
            
            dataset['images'].append(img_info)
            
            # 按事件组织标注
            if event_name not in dataset['annotations']:
                dataset['annotations'][event_name] = []
            dataset['annotations'][event_name].append({
                'img_name': img_name,
                'face_boxes': face_boxes,
                'difficulty': {
                    'easy': is_easy,
                    'medium': is_medium,
                    'hard': is_hard
                }
            })
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存JSON文件
    with open(os.path.join(output_dir, 'wider_face_val.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)

if __name__ == '__main__':
    gt_dir = 'ground_truth'  # mat文件目录
    output_dir = 'json_annotations'  # 输出目录
    mat2json(gt_dir, output_dir)