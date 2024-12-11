import os.path
import torch
import torch.utils.data as data
import cv2
import numpy as np

"""
WiderFace数据集加载模块
主要功能:
    - 加载WiderFace数据集的图像和标注
    - 支持训练和验证数据的读取
    - 处理人脸边界框和关键点标注

数据格式:
    - 图像路径存储在self.imgs_path
    - 标注信息存储在self.words
    - 每个标注包含:边界框坐标和关键点坐标
"""

class WiderFaceDetection(data.Dataset):
    """
    WiderFace数据集类
    Args:
        txt_path: 标注文件路径
        preproc: 预处理函数
    """
    def __init__(self, txt_path, preproc=None):
        """
        初始化数据集
        处理标注文件，提取图像路径和标注信息
        标注格式:
            # 图像路径
            x1 y1 w h x1_kp y1_kp ... x5_kp y5_kp  # 每行一个人脸标注
        """
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        """
        获取单个样本
        Returns:
            img: 预处理后的图像张量
            target: 标注信息，包含边界框和关键点
        """
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

def detection_collate(batch):
    """
    数据批处理函数
    处理不同图像中目标数量不同的情况
    Args:
        batch: 批量数据
    Returns:
        imgs: 批量图像张量 [B,C,H,W]
        targets: 批量标注信息列表
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
