from scipy.io import loadmat, savemat
import os
import numpy as np
from PIL import Image
import tqdm

def create_front_camera_gt(gt_dir, images_root):
    """创建前置相机场景的ground truth列表
    
    参数:
        gt_dir: 包含wider_face_val.mat的目录路径
        images_root: WIDER Face验证集图片根目录,包含各个event子目录
    """
    # 加载原始标注文件
    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    
    # 获取基本信息
    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']
    
    # 创建新的gt_list
    front_camera_gt = []
    
    # 使用tqdm显示处理进度
    pbar = tqdm.tqdm(range(len(event_list)))
    
    # 创建前置相机图片保存目录
    front_camera_images_root = os.path.join(os.path.dirname(images_root), 'front_camera_images')
    os.makedirs(front_camera_images_root, exist_ok=True)
    
    # 创建标注文件
    annotation_file = os.path.join(front_camera_images_root, 'front_camera_annotations.txt')
    annotations = []
    
    # 遍历每个event
    for event_idx in pbar:
        event_gt = []
        event_name = str(event_list[event_idx][0][0])
        img_list = file_list[event_idx][0]
        bbx_list = facebox_list[event_idx][0]
        
        # 创建对应的event目录
        event_output_dir = os.path.join(front_camera_images_root, event_name)
        os.makedirs(event_output_dir, exist_ok=True)
        
        pbar.set_description(f'处理事件: {event_name}')
        
        # 遍历event中的每张图片
        for img_idx in range(len(img_list)):
            boxes = bbx_list[img_idx][0]
            if len(boxes) == 0:
                event_gt.append([np.array([])])
                continue
            
            # 读取实际图片获取尺寸
            img_name = str(img_list[img_idx][0][0]) + '.jpg'
            img_path = os.path.join(images_root, event_name, img_name)
            
            try:
                img = Image.open(img_path)
                img_width, img_height = img.size
                img_area = img_width * img_height
                
                # 计算每个人脸框的面积占比
                valid_idx = []
                has_large_face = False
                for box_idx, box in enumerate(boxes):
                    x, y, w, h = box
                    face_area = w * h
                    area_ratio = face_area / img_area
                    
                    if area_ratio > 0.2:  # 筛选占比大于20%的框
                        valid_idx.append(box_idx + 1)  # 索引从1开始
                        has_large_face = True
                
                # 如果有大尺寸人脸，复制图片到新目录并记录标注
                if has_large_face:
                    dst_path = os.path.join(event_output_dir, img_name)
                    try:
                        # 复制图片前确保图片对象是打开的
                        img.copy().save(dst_path)
                        
                        # 准备标注信息
                        relative_path = os.path.join(event_name, img_name)
                        annotation_lines = [
                            relative_path,  # 图片路径
                            str(len(boxes)),  # 人脸数量
                        ]
                        # 添加每个人脸框的坐标
                        for box in boxes:
                            x, y, w, h = box
                            annotation_lines.append(f"{x} {y} {w} {h}")
                        
                        # 将该图片的标注信息加入列表
                        annotations.extend(annotation_lines)
                        
                    except Exception as e:
                        print(f"警告: 无法处理图片 {dst_path}, 错误: {e}")
                
                event_gt.append([np.array(valid_idx)])
                
            except Exception as e:
                print(f"警告: 无法读取图片 {img_path}, 错误: {e}")
                event_gt.append([np.array([])])
                continue
            
            finally:
                # 确保图片对象被正确关闭
                if 'img' in locals():
                    img.close()
        
        front_camera_gt.append([event_gt])
    
    # 写入标注文件
    try:
        with open(annotation_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(annotations))
        print(f"\n已生成标注文件: {annotation_file}")
    except Exception as e:
        print(f"警告: 无法写入标注文件, 错误: {e}")
    
    # 保存为.mat文件
    output_path = os.path.join(gt_dir, 'wider_front_camera_val.mat')
    savemat(output_path, {'gt_list': np.array(front_camera_gt, dtype=object)})
    print(f"\n已生成前置相机评估文件: {output_path}")

if __name__ == '__main__':
    # WIDER Face数据集验证集图片目录
    images_root = "C:/Project/ai/cv-assignment/4. final/dataset/wilderface_val/images"
    # ground truth文件目录
    gt_dir = "C:/Project/ai/cv-assignment/4. final/eval/widerface_evaluate/ground_truth"
    
    create_front_camera_gt(gt_dir, images_root)