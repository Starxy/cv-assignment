from dataset import load_data_blood_cell
import os
import torch
from ssd import TinySSD
from train import train

def main():
    # 切换到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 加载数据集
    batch_size = 32
    train_iter, val_iter = load_data_blood_cell(batch_size,data_dir='./blood_cell_detection')

    # 设置基本参数
    num_classes = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型和数据加载器
    net = TinySSD(num_classes=num_classes)
    # 训练模型
    train(net, train_iter, val_iter, 
          num_epochs=20, 
          lr=0.2, 
          device=device,
          title="SSD Object Detection")

if __name__ == '__main__':
    main()
