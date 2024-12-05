from dataset import load_data_blood_cell
import os
import torch
from ssd import TinySSD
from train import train

def main():
    """主函数 - 仅用于模型训练"""
    # 切换到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 加载数据集
    batch_size = 32
    train_iter, val_iter = load_data_blood_cell(batch_size, data_dir='./blood_cell_detection')

    # 设置基本参数
    num_classes = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型并训练
    net = TinySSD(num_classes=num_classes)
    train(net, train_iter, val_iter, 
          num_epochs=50, 
          lr=0.1, 
          device=device,
          title="SSD Object Detection")

if __name__ == '__main__':
    main()
