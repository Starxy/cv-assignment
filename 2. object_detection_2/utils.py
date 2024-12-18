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
        
        
        # 使用constrained_layout自动调整布局
        self.fig.set_constrained_layout(True)
        
        # 将标题转换为文件名格式（小写+下划线）
        # 获取当前时间戳
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # 构建文件名,包含时间戳
        save_path = f"{self.title.lower().replace(' ', '_')}_{timestamp}.png"
        self.fig.savefig(save_path, dpi=300, bbox_inches='tight')
