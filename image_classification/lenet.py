"""
LeNet-5 模型实现

@Author: DONG Jixing
@Date: 2024-11-10
"""
import torch
from torch import nn

class LeNet(nn.Module):
    """LeNet-5卷积神经网络架构"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # 第一个卷积层 (1, 28, 28) -> (6, 24, 24)
            nn.Conv2d(1, 6, kernel_size=5, padding=0),
            nn.ReLU(),
            # 第一个池化层 (6, 24, 24) -> (6, 12, 12)
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积层 (6, 12, 12) -> (16, 8, 8)
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            # 第二个池化层 (16, 8, 8) -> (16, 4, 4)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc = nn.Sequential(
            # 全连接层
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
        
        # 初始化模型权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
