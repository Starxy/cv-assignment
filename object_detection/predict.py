import torch
import os
import random
from torchvision.io import read_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ssd import TinySSD
import argparse
from train import box_corner_to_center, box_center_to_corner, box_iou


def get_latest_model_path(checkpoints_dir='checkpoints'):
    """获取最新的模型文件路径"""
    if not os.path.exists(checkpoints_dir):
        return None
        
    model_files = [f for f in os.listdir(checkpoints_dir) 
                  if f.startswith('ssd_model_') and f.endswith('.pth')]
    
    if not model_files:
        return None
        
    latest_model = sorted(model_files)[-1]
    return os.path.join(checkpoints_dir, latest_model)

def load_trained_model(model_path, num_classes, device):
    """加载训练好的模型"""
    net = TinySSD(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net = net.to(device)
    net.eval()
    return net

def run_prediction(test_dir, num_classes=4, device=None, model_path=None):
    """运行预测流程
    
    Args:
        test_dir: 测试图片目录
        num_classes: 类别数量
        device: 运行设备
        model_path: 指定模型路径，如果为None则使用最新的模型
    """
    # 强制使用CPU设备
    device = torch.device('cpu')
    
    # 如果没有指定模型路径，则获取最新的模型
    if model_path is None:
        model_path = get_latest_model_path()
    
    if model_path and os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        trained_net = load_trained_model(model_path, num_classes, device)
        predict_and_visualize(trained_net, test_dir, device)
    else:
        print("未找到训练好的模型文件")

def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框"""
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox

def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序"""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 保留预测边界框的指标
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)
def predict_and_visualize(model, img_dir, device, conf_threshold=0.5, iou_threshold=0.5):
    """从测试目录随机选择一张图片进行预测并可视化
    
    这个函数的主要功能是:
    1. 从指定目录随机选择一张图片
    2. 使用模型进行目标检测预测
    3. 可视化检测结果
    
    Args:
        model: 训练好的目标检测模型
        img_dir: 测试图片所在目录
        device: 运行设备(CPU/GPU)
        conf_threshold: 置信度阈值,默认0.5
        iou_threshold: IoU阈值,默认0.5
    """
    # 将模型设置为评估模式
    model.eval()
    
    # 从目录中筛选出所有图片文件并随机选择一张
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    img_file = random.choice(img_files)
    print(f"处理图片: {img_file}")
    
    # 读取图片并进行预处理
    # 不需要除以255是因为read_image函数已经将像素值标准化到[0,1]范围
    img_path = os.path.join(img_dir, img_file)
    image = read_image(img_path).float()
    X = image.unsqueeze(0).to(device)  # 添加batch维度并移至指定设备
    
    # 定义类别名称和对应的显示颜色
    class_names = ['Background', 'Platelets', 'RBC', 'WBC']  # 包含背景类
    colors = ['gray', 'r', 'g', 'b']  # 对应每个类别的颜色
    
    # 使用模型进行预测
    model.eval()  # 确保模型处于评估模式
    anchors, cls_preds, bbox_preds = model(X.to(device))
    # 将分类预测转换为概率并调整维度顺序
    cls_probs = torch.softmax(cls_preds, dim=2).permute(0, 2, 1)
    # 使用非极大值抑制进行后处理
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    # 过滤掉背景类的预测结果(-1表示背景)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    output2 = output[0, idx]
    print("检测结果:")
    print(output2)
    
    # 保存结果
    save_dir = 'predictions'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'prediction_{os.path.splitext(img_file)[0]}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"预测结果已保存到 {save_path}")

def main():
    """预测脚本的主函数"""
    parser = argparse.ArgumentParser(description='血细胞检测模型预测程序')
    parser.add_argument('--test_dir', type=str, default='./blood_cell_detection/valid',
                      help='测试图片目录路径')
    parser.add_argument('--num_classes', type=int, default=4,
                      help='类别数量')
    parser.add_argument('--model_path', type=str, default=None,
                      help='模型文件路径，默认使用最新的模型')
    
    args = parser.parse_args()
    
    # 切换到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 运行预测
    run_prediction(
        test_dir=args.test_dir,
        num_classes=args.num_classes,
        model_path=args.model_path
    )

if __name__ == '__main__':
    main() 