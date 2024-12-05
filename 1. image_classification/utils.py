"""
工具函数模块

@Author: DONG Jixing
@Date: 2024-11-10
"""

import torch
from torch import nn
import time
import matplotlib.pyplot as plt
import numpy as np

class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(6, 4), title=None):
        """Defined in :numref:`sec_utils`"""
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize,
                                         constrained_layout=True)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 如果有标题，立即设置
        if title:
            self.fig.suptitle(title, fontsize=12, y=0.95)
        # 使用lambda函数捕获参数
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        self.title = title  # 保存标题
        self.results = None  # 添加用于存储注释文字的属性

    def add(self, x, y):
        """向图表中添加多个数据点"""
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        plt.draw()
        plt.pause(0.1)

    def set_axes(self, axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """设置matplotlib的轴"""
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def set_hyperparameters(self, hyperparameters):
        """设置要在图表上显示的超参数"""
        self.hyperparameters = hyperparameters

    def set_results(self, results):
        """设置要在图表上显示的注释文字"""
        self.results = results

    def save(self):
        """保存图表到文件"""
        if self.title is None:
            return
            
        # 如果有注释文字，添加到图表上
        if self.results:
            self.axes[0].text(
                0.98, 0.02, self.results,
                transform=self.axes[0].transAxes,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                fontsize=8
            )
        
        # 如果有超参数，添加到图表上
        if self.hyperparameters:
            # 将超参数字典转换为字符串格式
            hyperparams_str = f"Batch Size={self.hyperparameters.get('bs', 'N/A')}\n" \
                            f"Learning Rate={self.hyperparameters.get('lr', 'N/A')}"
            self.axes[0].text(
                0.02, 0.02, hyperparams_str,
                transform=self.axes[0].transAxes,
                verticalalignment='bottom',
                horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                fontsize=8
            )
        
        # 使用constrained_layout自动调整布局
        self.fig.set_constrained_layout(True)
        
        # 将标题转换为文件名格式（小写+下划线）
        # 获取当前时间戳
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # 构建文件名,包含bs、lr和时间戳
        save_path = f"{self.title.lower().replace(' ', '_')}_bs{self.hyperparameters.get('bs', 'NA')}_lr{self.hyperparameters.get('lr', 'NA')}_{timestamp}.png"
        self.fig.savefig(save_path, dpi=300, bbox_inches='tight')

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    # 如果y_hat是二维张量(矩阵)且第二维大于1,说明是分类问题的预测概率分布
    # 需要取每行最大值的索引作为预测类别
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    
    # 将预测值的数据类型转换为标签y的类型,然后判断是否相等
    # 返回一个由0(不相等)和1(相等)组成的张量
    cmp = y_hat.type(y.dtype) == y
    
    # 将比较结果转换为y的数据类型并求和,得到预测正确的样本数
    # 最后转换为Python标量返回
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train(net, train_iter, test_iter, num_epochs, lr, device, title=None):
    """训练模型的主函数
    
    Args:
        net: 要训练的神经网络模型
        train_iter: 训练数据集迭代器
        test_iter: 测试数据集迭代器 
        num_epochs: 训练轮数
        lr: 学习率
        device: 训练设备(CPU/GPU)
        title: 图表标题，默认使用模型类名
    """

    print('training on', device)
    net.to(device)  # 将模型移至指定设备
    
    # 定义优化器和损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    
    # 创建动画器用于可视化训练过程
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'],
                        title=title if title else net.__class__.__name__)  # 如果没有提供标题则使用模型类名
    # 设置超参数信息到动画器
    animator.set_hyperparameters({
        'lr': lr,
        'bs': train_iter.batch_size,
    })
    # 初始化计时器和批次数
    timer, num_batches = Timer(), len(train_iter)
    
    # 开始训练循环
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = Accumulator(3)
        net.train()  # 设置为训练模式
        
        # 遍历训练数据集
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()  # 清除梯度
            
            # 将数据移至指定设备
            X, y = X.to(device), y.to(device)
            
            # 前向传播
            y_hat = net(X)
            l = loss(y_hat, y)
            
            # 反向传播和优化
            l.backward()
            optimizer.step()
            
            # 记录训练指标
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            
            # 计算当前批次的训练损失和准确率
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            
            # 更新动画显示
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        
        # 在测试集上评估模型
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    
    # 在打印最终结果之前保存图表
    animator.set_results(f'Final Results:\nLoss: {train_l:.3f}\nTrain Acc: {train_acc:.3f}\nTest Acc: {test_acc:.3f}')
    animator.save()
    
    # 打印最终训练结果
    print(f'{title} with learning rate {lr} batch size {train_iter.batch_size} Final Results:')
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')