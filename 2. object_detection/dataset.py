import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from pathlib import Path

class BloodCellDataset(Dataset):
    """血细胞检测数据集的自定义数据集类"""
    def __init__(self, json_file, img_dir, is_train=True, max_objects=10):
        """
        Args:
            json_file: COCO格式标注文件的路径
            img_dir: 图片目录的路径
            is_train: 是否为训练集
            max_objects: 每张图片最大目标数量
        """
        self.img_dir = img_dir
        # 加载COCO格式的标注
        with open(json_file, 'r') as f:
            self.coco = json.load(f)
            
        self.images = self.coco['images']
        self.annotations = self.coco['annotations']
        
        # 建立图片ID到标注的映射
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
        print(f'read {len(self.images)} {"training" if is_train else "validation"} examples')
        self.max_objects = max_objects

    def __getitem__(self, idx):
        # 获取图片信息
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # 读取图片
        image = read_image(img_path).float() / 255.0
        
        # 获取该图片的所有标注
        img_id = img_info['id']  # 修改这里：使用'id'而不是'img_id'
        anns = self.img_to_anns.get(img_id, [])
        
        # 转换标注格式为 [class_id, xmin, ymin, xmax, ymax]
        targets = []
        for ann in anns:
            bbox = ann['bbox']  # COCO格式为 [x,y,width,height]
            # 转换为 [xmin,ymin,xmax,ymax] 格式
            xmin = bbox[0] / img_info['width']
            ymin = bbox[1] / img_info['height']
            xmax = (bbox[0] + bbox[2]) / img_info['width']
            ymax = (bbox[1] + bbox[3]) / img_info['height']
            
            targets.append([
                ann['category_id'],  # 类别ID
                xmin, ymin, xmax, ymax
            ])
            
        # 如果没有标注,添加一个填充标注
        if not targets:
            targets.append([-1, 0, 0, 0, 0])
            
        if len(targets) == 0:
            # 如果没有标注，填充max_objects个无效标注
            targets = [[-1, 0, 0, 0, 0]] * self.max_objects
        else:
            # 如果标注数量不足max_objects，用无效标注填充
            if len(targets) < self.max_objects:
                padding = [[-1, 0, 0, 0, 0]] * (self.max_objects - len(targets))
                targets.extend(padding)
            # 如果标注数量超过max_objects，截断
            elif len(targets) > self.max_objects:
                targets = targets[:self.max_objects]
            
        return image, torch.tensor(targets)

    def __len__(self):
        return len(self.images)

def load_data_blood_cell(batch_size, data_dir='object_detection/blood_cell_detection'):
    """加载血细胞检测数据集
    
    Args:
        batch_size: 批量大小
        data_dir: 数据集根目录
    Returns:
        train_iter: 训练集的数据加载器
        val_iter: 验证集的数据加载器
    """
    train_dataset = BloodCellDataset(
        os.path.join(data_dir, 'train/_annotations.coco.json'),
        os.path.join(data_dir, 'train'),
        is_train=True
    )
    
    val_dataset = BloodCellDataset(
        os.path.join(data_dir, 'valid/_annotations.coco.json'),
        os.path.join(data_dir, 'valid'),
        is_train=False
    )
    
    train_iter = DataLoader(train_dataset, batch_size=batch_size, 
                          shuffle=True, num_workers=2)
    val_iter = DataLoader(val_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=2)
    
    return train_iter, val_iter
