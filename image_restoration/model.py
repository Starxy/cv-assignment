import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


class ConvBlockLeakyRelu(nn.Module):
    '''
    包含一个Conv2d层和LeakyReLU激活函数的基础卷积块
    '''

    def __init__(self, chanel_in, chanel_out, kernel_size, stride=1, padding=1):
        super(ConvBlockLeakyRelu, self).__init__()
        self.block = nn.Sequential(
            # 二维卷积层，不使用偏置项
            nn.Conv2d(chanel_in, chanel_out, kernel_size, stride=stride, padding=padding, bias=False),
            # LeakyReLU激活函数，负半轴斜率为0.1
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 编码器部分 - 通过卷积和池化逐步降低特征图尺寸
        self.enc_conv01 = nn.Sequential(
            # 第一层：3通道输入，48通道输出
            ConvBlockLeakyRelu(3, 48, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1),
            nn.MaxPool2d(2)  # 2x2最大池化，特征图尺寸减半
        )

        self.enc_conv2 = nn.Sequential(
            ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )

        self.enc_conv3 = nn.Sequential(
            ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )

        self.enc_conv4 = nn.Sequential(
            ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )

        self.enc_conv56 = nn.Sequential(
            ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1),
            nn.MaxPool2d(2),
            ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1),
        )

        # 解码器部分 - 通过上采样和拼接恢复图像尺寸
        self.dec_conv5ab = nn.Sequential(
            # 96通道输入（48+48，来自跳跃连接），96通道输出
            ConvBlockLeakyRelu(96, 96, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(96, 96, 3, stride=1, padding=1),
        )

        self.dec_conv4ab = nn.Sequential(
            ConvBlockLeakyRelu(144, 96, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(96, 96, 3, stride=1, padding=1),
        )

        self.dec_conv3ab = nn.Sequential(
            ConvBlockLeakyRelu(144, 96, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(96, 96, 3, stride=1, padding=1),
        )

        self.dec_conv2ab = nn.Sequential(
            ConvBlockLeakyRelu(144, 96, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(96, 96, 3, stride=1, padding=1),
        )

        self.dec_conv1abc = nn.Sequential(
            ConvBlockLeakyRelu(99, 64, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(64, 32, 3, stride=1, padding=1),
            nn.Conv2d(32, 3, 3, stride=1, padding=1, bias=True)
        )

    def forward(self, x):
        # 存储编码器各层输出，用于跳跃连接
        residual_connection = [x]

        # 编码过程
        x = self.enc_conv01(x)
        residual_connection.append(x)

        x = self.enc_conv2(x)
        residual_connection.append(x)

        x = self.enc_conv3(x)
        residual_connection.append(x)

        x = self.enc_conv4(x)
        residual_connection.append(x)

        x = self.enc_conv56(x)

        # 解码过程
        # 每一步都包含：上采样 -> 特征拼接 -> 卷积处理
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, residual_connection.pop()], dim=1)
        x = self.dec_conv5ab(x)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, residual_connection.pop()], dim=1)
        x = self.dec_conv4ab(x)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, residual_connection.pop()], dim=1)
        x = self.dec_conv3ab(x)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, residual_connection.pop()], dim=1)
        x = self.dec_conv2ab(x)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, residual_connection.pop()], dim=1)
        x = self.dec_conv1abc(x)

        return x


class Dataset(torch.utils.data.Dataset):
    '''
    自定义数据集类，用于数据增强
    支持水平和垂直翻转，以及输入-目标对的随机交换
    '''

    def __init__(self, dataset1, dataset2):
        # 将输入和目标图像堆叠在一起
        self.datasets = torch.cat([dataset1[:, None], dataset2[:, None]], dim=1)
        # 定义数据增强变换
        self.transforms = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(0.5),  # 50%概率水平翻转
            transforms.RandomVerticalFlip(0.5),    # 50%概率垂直翻转
        )

    def __getitem__(self, i):
        if torch.rand(1) > 0.5:
            return self.transforms(self.datasets[i])
        else:
            return self.transforms(self.datasets[i, [1, 0]])

    def __len__(self):
        return len(self.datasets)


class Model():
    '''
    完整的模型类，包含训练和预测功能
    '''
    def __init__(self) -> None:
        # 选择设备（GPU或CPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化网络、优化器和损失函数
        self.net = Net()
        self.net.to(self.device)
        # 使用SGD优化器，学习率0.001，动量0.8
        self.optim = torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.8)
        # 使用均方误差损失
        self.loss = nn.MSELoss()
        # 添加loss记录列表
        self.loss_history = []

    def load_pretrained_model(self, path) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))

    def train(self, train_input, train_target, num_epochs, batch_size=8, num_workers=2) -> None:
        '''
        训练模型，使用TQDM进度条显示训练进度
        :param train_input: 输入图像张量
        :param train_target: 目标图像张量 
        :param num_epochs: 训练轮数
        :param batch_size: 批次大小
        :param num_workers: 数据加载的工作进程数
        '''

        train_loader = torch.utils.data.DataLoader(
            Dataset(
                train_input,
                train_target
            ),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)

        self.net.train()

        for epoch in range(0, num_epochs):
            loop = tqdm(train_loader)
            train_loss = []
            for i, data in enumerate(loop):
                source, target = data[:, 0].float().cuda() / 255, data[:, 1].float().cuda() / 255
                denoised = self.net(source)

                loss_ = self.loss(denoised, target)
                train_loss.append(loss_.detach().cpu().item())

                # 记录每个batch的loss
                self.loss_history.append(loss_.detach().cpu().item())

                self.optim.zero_grad()
                loss_.backward()
                self.optim.step()
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(loss=np.mean(train_loss))

        # 训练结束后绘制loss曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.xlabel('total batches')
        plt.ylabel('loss')
        plt.grid(True)
        plt.savefig('training_loss.png')
        plt.close()

    def predict(self, test_input) -> torch.Tensor:
        '''
        对输入图像进行去噪
        输入：(N1, C, H, W)大小的张量，像素值在0-255之间
        输出：去噪后的图像，像素值已归一化到0-255之间
        '''
        self.net.eval()

        def normalization_cut(imgs):
            '''
            将像素值裁剪到[0,1]范围内
            '''
            imgs_shape = imgs.shape
            imgs = imgs.flatten()
            imgs[imgs < 0] = 0
            imgs[imgs > 1] = 1
            imgs = imgs.reshape(imgs_shape)
            return imgs

        return 255 * normalization_cut(self.net((test_input / 255).to(self.device)))

    def save(self, path):
        torch.save(self.net.state_dict(), path)
