"""
主程序入口

@Author: DONG Jixing
@Date: 2024-11-10
"""
import torch
import os
from dataset_loader import DatasetLoader
from lenet import LeNet
from resnet import ResNet
from utils import train

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main():
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集
    dataset_loader = DatasetLoader(
        dataset_name='FashionMNIST',
        batch_size=256,
        root='./data',
        num_workers=4
    )
    train_loader, test_loader = dataset_loader.get_data_loaders()
    
    # 设置训练参数
    num_epochs = 10

    # 尝试不同的学习率
    learning_rates = [0.1, 0.05, 0.01] 
    for lr in learning_rates:
        # 初始化所有模型
        lenet = LeNet()
        resnet18 = ResNet(layers=[2, 2, 2, 2], num_classes=10)
        resnet34 = ResNet(layers=[3, 4, 6, 3], num_classes=10)
   
        print(f"\n使用学习率: {lr}")
        # 训练LeNet模型
        train(lenet, train_loader, test_loader, num_epochs, lr=lr, device=device, 
              title="LeNet on Fashion-MNIST")
        
        # 训练ResNet18模型
        train(resnet18, train_loader, test_loader, num_epochs, lr=lr, device=device, 
              title="ResNet18 on Fashion-MNIST")
        
        # 训练ResNet34模型
        train(resnet34, train_loader, test_loader, num_epochs, lr=lr, device=device, 
              title="ResNet34 on Fashion-MNIST")

if __name__ == '__main__':
    main()
