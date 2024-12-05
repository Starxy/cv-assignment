import torch 
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from ssd import TinySSD
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

def predict(net, X, device):
    # 将模型设置为评估模式
    net.eval()
    
    # 通过网络获取锚框、类别预测和边界框预测
    # anchors: 生成的锚框
    # cls_preds: 每个锚框的类别预测
    # bbox_preds: 每个锚框的边界框偏移量预测
    anchors, cls_preds, bbox_preds = net(X.to(device))
    
    # 对类别预测进行softmax得到概率分布,并调整维度顺序
    # permute(0,2,1)将维度从[batch,anchors,classes]变为[batch,classes,anchors]
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    
    # 使用multibox_detection将预测结果转换为边界框
    # 返回格式:[batch, num_anchors, 6]
    # 6个值分别是:[类别标签, 置信度, xmin, ymin, xmax, ymax]
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    # 筛选出有效的预测框(类别标签不为-1)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    
    # 返回第一张图片的有效预测结果
    return output[0, idx]

def display(img, output, threshold, save_path):
    # 将tensor转换为PIL Image对象
    img_np = img.numpy().astype('uint8')
    img_pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img_pil)
    
    # 尝试加载中文字体，如果失败则使用默认字体
    try:
        font = ImageFont.truetype("simhei.ttf", 20)  # 使用黑体，大小20
    except:
        font = ImageFont.load_default()
    
    # 遍历每个检测框的结果
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
            
        # 获取图像的高度和宽度
        h, w = img.shape[0:2]
        # 将相对坐标转换为绝对坐标
        bbox = row[2:6] * torch.tensor((w, h, w, h), device=row.device)
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        
        # 绘制矩形框
        draw.rectangle([(xmin, ymin), (xmax, ymax)], 
                      outline='red', width=2)
        
        # 绘制置信度分数
        text = f'{score:.2f}'
        # 计算文本大小以确定背景矩形的尺寸
        text_bbox = draw.textbbox((xmin, ymin), text, font=font)
        # 绘制文本背景
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                      fill='red')
        # 绘制白色文本
        draw.text((xmin, ymin), text, fill='white', font=font)
    
    # 保存图片
    img_pil.save(save_path)

def load_model(model_path, num_classes=1):
    """加载训练好的模型
    
    Args:
        model_path: 模型权重文件路径
        num_classes: 类别数量
        
    Returns:
        加载了权重的模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = TinySSD(num_classes=num_classes)
    
    # 加载检查点
    try:
        checkpoint = torch.load(model_path, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载模型检查点，训练轮次: {checkpoint['epoch']}")
        print(f"训练时的分类错误率: {checkpoint['train_cls_err']:.4f}")
        print(f"训练时的边界框MAE: {checkpoint['train_bbox_mae']:.4f}")
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return None, None
    
    net.to(device)
    return net, device

def combine_predictions(pred_dir, output_path, images_per_row=4):
    """将预测结果图像拼接成组合图
    
    Args:
        pred_dir: 预测结果图片所在目录
        output_path: 组合图片保存路径
        images_per_row: 每行图片数量
    """
    # 获取所有预测结果图片
    pred_images = [f for f in os.listdir(pred_dir) if f.startswith('pred_')]
    
    if not pred_images:
        print("没有找到预测结果图片")
        return
        
    # 读取第一张图片来获取图片尺寸
    first_img = Image.open(os.path.join(pred_dir, pred_images[0]))
    img_width, img_height = first_img.size
    
    # 每组8张图片，所以高度固定为2行
    grid_width = images_per_row * img_width
    grid_height = 2 * img_height  # 固定2行
    
    # 创建空白画布
    grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # 拼接图片
    for idx, img_name in enumerate(pred_images):
        img = Image.open(os.path.join(pred_dir, img_name))
        # 在8张图片的组内计算行列位置
        group_idx = idx % 8
        row = group_idx // images_per_row  # 0 或 1
        col = group_idx % images_per_row   # 0 到 3
        grid_img.paste(img, (col * img_width, row * img_height))
        
        # 每8张图片保存一次，或者是最后一组
        if (idx + 1) % 8 == 0 or idx == len(pred_images) - 1:
            # 如果是最后一组且图片数量不足8张
            if idx == len(pred_images) - 1 and (idx + 1) % 8 != 0:
                # 计算实际使用的行数
                used_rows = (group_idx // images_per_row) + 1
                grid_height = used_rows * img_height
                # 裁剪到实际使用的区域
                grid_img = grid_img.crop((0, 0, grid_width, grid_height))
            
            # 生成保存路径
            save_name = f'combined_predictions_{(idx + 1) // 8}.jpg'
            save_path = os.path.join(os.path.dirname(output_path), save_name)
            grid_img.save(save_path, quality=95)
            print(f"已保存组合图片: {save_path}")
            # 创建新的空白画布
            grid_img = Image.new('RGB', (grid_width, grid_height), color='white')

def main():
    # 切换到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 设置模型路径和图片路径    
    checkpoints_dir = 'checkpoints'
    
    # 获取最新的模型文件
    model_files = [f for f in os.listdir(checkpoints_dir) if f.startswith('ssd_model_')]
    if not model_files:
        print("未找到训练好的模型文件！")
        return
    
    latest_model = sorted(model_files)[-1]  # 获取最新的模型文件
    model_path = os.path.join(checkpoints_dir, latest_model)
    
    # 加载模型
    net, device = load_model(model_path)
    print(f"已加载模型: {model_path}")
    # 设置测试图片目录
    test_dir = 'bananas_val/images'  # 假设验证集图片在这个目录
    if not os.path.exists(test_dir):
        print(f"测试图片目录 {test_dir} 不存在！")
        return
        
    # 创建输出目录
    output_dir = 'predictions'
    os.makedirs(output_dir, exist_ok=True)
    
    # 对测试集中的每张图片进行预测
    for img_file in os.listdir(test_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(test_dir, img_file)
            print(f"正在处理图片: {img_path}")
            
            # 读取图像并进行预处理
            X = torchvision.io.read_image(img_path).unsqueeze(0).float()
            img = X.squeeze(0).permute(1, 2, 0).long()
            # 执行预测
            output = predict(net, X, device)
            # 设置保存路径并显示结果
            save_path = os.path.join(output_dir, f'pred_{img_file}')
            display(img, output.cpu(), threshold=0.8, save_path=save_path)
            print(f"预测结果已保存到: {save_path}")
    
    # 在处理完所有预测后，添加以下代码
    print("正在生成组合预测图...")
    combine_predictions(output_dir, os.path.join(output_dir, 'combined_predictions.jpg'))

if __name__ == '__main__':
    # main()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    output_dir = 'predictions'
    combine_predictions(output_dir, os.path.join(output_dir, 'combined_predictions.jpg'))
