"""
ResNet 模型实现

@Author: DONG Jixing
@Date: 2024-11-10
"""
import torch
from torch import nn

class ResidualBlock(nn.Module):
    """ResNet的基本残差块
    
    这是ResNet中的基本构建块,包含两个3x3卷积层和一个shortcut连接。
    如果输入和输出维度不匹配,会通过1x1卷积进行调整。
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数 
        stride (int): 步长,用于下采样,默认为1
    """
    expansion = 1  # 输出通道数相对于输入通道数的倍增系数
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 第一个卷积层: 3x3卷积,可能改变特征图大小和通道数
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # 批量归一化层
        
        # 第二个卷积层: 3x3卷积,保持特征图大小和通道数不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)  # 批量归一化层
        
        # shortcut连接: 用于将输入直接加到输出上
        self.shortcut = nn.Sequential()  # 默认为恒等映射
        # 当步长不为1或通道数改变时,需要调整shortcut分支的维度
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 1x1卷积用于调整维度
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x  # 保存原始输入用于shortcut连接
        
        # 第一个卷积块: 卷积+BN+ReLU
        out = torch.relu(self.bn1(self.conv1(x)))
        
        # 第二个卷积块: 卷积+BN(注意这里没有ReLU)
        out = self.bn2(self.conv2(out))
        
        # 残差连接: 将shortcut分支的结果加到主分支上
        out += self.shortcut(identity)
        return torch.relu(out)  # 最后再通过ReLU激活

class ResNet(nn.Module):
    """ResNet模型
    
    这是ResNet的主体架构,包含一个初始卷积层和4个残差层组。每个残差层组包含多个残差块,
    用于逐步提取图像特征。最后通过全局平均池化和全连接层完成分类。
    
    Args:
        block (nn.Module): 残差块类型,默认为ResidualBlock
        layers (List[int]): 每个残差层组包含的残差块数量,默认为[2,2,2,2]
        num_classes (int): 分类类别数,默认为10
    """
    def __init__(self, block=ResidualBlock, layers=[2, 2, 2, 2], num_classes=10):
        super().__init__()

        # 残差块类型
        self.block = block
        
        # 初始卷积层: 7x7卷积,步长为2
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)  # 批量归一化
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化
        
        # 4个残差层组,通道数依次增加,空间分辨率依次减半
        self.in_channels = 64  # 初始输入通道数,用于在构建跟踪当前特征图的通道数
        self.layer1 = self._make_layer(64, layers[0], stride=1)   # 输出: 64通道
        self.layer2 = self._make_layer(128, layers[1], stride=2)  # 输出: 128通道
        self.layer3 = self._make_layer(256, layers[2], stride=2)  # 输出: 256通道
        self.layer4 = self._make_layer(512, layers[3], stride=2)  # 输出: 512通道
        
        # 分类头: 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化到1x1
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接分类层
        
        # 初始化模型权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用Xavier初始化卷积层
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                # 初始化BN层的gamma为1,beta为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, out_channels, blocks, stride=1):
        """构建残差层组
        
        Args:
            block: 残差块类型
            out_channels: 输出通道数
            blocks: 残差块数量
            stride: 第一个残差块的步长,用于下采样
            
        Returns:
            nn.Sequential: 残差层组
        """
        layers = []
        # 第一个残差块可能需要调整维度
        layers.append(self.block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * self.block.expansion
        # 添加后续残差块
        for _ in range(1, blocks):
            layers.append(self.block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入图像张量,[B,1,H,W]
            
        Returns:
            输出预测张量,[B,num_classes]
        """
        # 初始层: 卷积+BN+ReLU+池化
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # 4个残差层组
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 分类层: 池化+展平+全连接
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def resnet18(num_classes=10):
    """返回ResNet-18模型"""
    return ResNet( [2, 2, 2, 2], num_classes)

def resnet34(num_classes=10):
    """返回ResNet-34模型"""
    return ResNet( [3, 4, 6, 3], num_classes)
