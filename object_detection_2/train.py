import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from ssd import TinySSD
import os
from datetime import datetime
from utils import Animator

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    """计算损失函数"""
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss = nn.L1Loss(reduction='none')
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    """分类精度"""
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    """边界框预测精度"""
    return float((torch.abs((bbox_preds - bbox_labels) * bbox_masks)).sum())

def train(net, train_iter, num_epochs, trainer, device, num_classes):
    """训练模型"""
    print('training on', device)
    net.to(device)
    
    # 创建保存目录
    os.makedirs('checkpoints', exist_ok=True)
    
    # 修改Animator以显示更多指标
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                          legend=['train cls error', 'train bbox mae'],
                          title='SSD Training')
    
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        net.train()
        for features, target in train_iter:
            timer.start()
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)
            anchors, cls_preds, bbox_preds = net(X)
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                         bbox_masks)
            l.mean().backward()
            trainer.step()
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                      bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                      bbox_labels.numel())
        
        # 计算训练指标
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        animator.add(epoch + 1, (cls_err, bbox_mae))
        
        # 打印当前epoch的训练信息
        print(f'epoch {epoch+1}, class err {cls_err:.3f}, bbox mae {bbox_mae:.3f}')
    
    # 保存训练结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = f'checkpoints/ssd_model_{timestamp}.pth'
    
    # 保存模型和训练状态
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': trainer.state_dict(),
        'epoch': num_epochs,
        'train_cls_err': cls_err,
        'train_bbox_mae': bbox_mae
    }, model_save_path)
    
    # 保存训练曲线
    animator.save()
    
    print(f'模型已保存至 {model_save_path}')
    print(f'训练曲线已保存至 checkpoints/training_curve_{timestamp}.png')
    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on {str(device)}')

def main():
    # 切换到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    batch_size = 32
    train_iter, _ = d2l.load_data_bananas(batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = TinySSD(num_classes=1)
    trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
    
    num_epochs = 20
    train(net, train_iter, num_epochs, trainer, device, num_classes=1)

if __name__ == '__main__':
    main()
