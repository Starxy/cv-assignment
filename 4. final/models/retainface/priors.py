import math
from itertools import product
import numpy as np

# RetinaFace的默认配置
RETINAFACE_CONFIG = {
    'min_sizes': [[16, 32], [64, 128], [256, 512]],  # 每个特征图层的先验框最小尺寸
    'steps': [8, 16, 32],  # 特征图相对于原图的步长
    'variance': [0.1, 0.2],  # 用于边界框编码和解码的方差
    'clip': False,  # 是否将先验框坐标裁剪到[0,1]范围内
}

def generate_priors(image_size, config=None):
    """
    生成RetinaFace的先验框
    Args:
        image_size: 输入图像尺寸,格式为(height, width)或int
        config: 配置字典,如果为None则使用默认配置
    Returns:
        numpy.ndarray: 形状为(N,4)的数组,每行表示一个先验框[cx,cy,w,h]
    """
    if config is None:
        config = RETINAFACE_CONFIG
        
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
        
    anchors = []
    # 计算每个特征图的高度和宽度
    feature_maps = [[
        math.ceil(image_size[0]/step), 
        math.ceil(image_size[1]/step)
    ] for step in config['steps']]
    
    # 遍历每个特征图层(不同分辨率的特征图用于检测不同大小的目标)
    for k, (map_height, map_width) in enumerate(feature_maps):
        step = config['steps'][k]
        # 遍历特征图的每个位置
        for i, j in product(range(map_height), range(map_width)):
            # 对每个最小尺寸生成先验框
            for min_size in config['min_sizes'][k]:
                # 计算先验框的宽高(相对值)
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                
                # 计算先验框的中心点坐标(相对值)
                dense_cx = [x * step / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * step / image_size[0] for y in [i + 0.5]]
                
                # 组合中心点坐标和尺寸,生成先验框
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]
                    
    # 将列表转换为数组并重塑维度
    output = np.array(anchors).reshape(-1, 4)
    if config['clip']:
        output = np.clip(output, 0, 1)  # 将坐标值限制在[0,1]范围内
        
    return output

def decode_boxes(loc, priors, variances=[0.1, 0.2]):
    """
    使用先验框解码预测的位置,以撤销训练时进行的偏移回归编码。

    参数:
        loc: 位置层的位置预测,形状: [num_priors, 4]
        priors: 中心偏移格式的先验框,形状: [num_priors, 4]
        variances: 先验框的方差

    返回:
        解码后的边界框预测
    """
    # 计算预测框的中心点
    cxcy = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]

    # 计算预测框的宽度和高度
    wh = priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])

    # 将中心点和尺寸转换为角点坐标
    boxes = np.empty_like(loc)
    boxes[:, :2] = cxcy - wh / 2  # xmin, ymin
    boxes[:, 2:] = cxcy + wh / 2  # xmax, ymax

    return boxes

def decode_landmarks(predictions, priors, variances=[0.1, 0.2]):
    """
    使用先验框解码预测的关键点,以撤销训练时进行的编码。

    参数:
        predictions: 定位层的关键点预测
            形状: [num_priors, 10],每个先验框包含 5 对关键点 (x, y)
        priors: 中心偏移格式的先验框
            形状: [num_priors, 4],每个先验框包含 (cx, cy, width, height)
        variances: 用于缩放解码值的先验框方差

    返回:
        解码后的关键点预测
            形状: [num_priors, 10],每行包含 5 个关键点的解码后的 (x, y) 坐标
    """
    # 将predictions重塑为[num_priors, 5, 2]以批量处理每对(x,y)坐标
    predictions = predictions.reshape(-1, 5, 2)
    
    # 扩展priors的维度以进行广播运算
    priors_xy = np.expand_dims(priors[:, :2], axis=1)  # [num_priors, 1, 2]
    priors_wh = np.expand_dims(priors[:, 2:], axis=1)  # [num_priors, 1, 2]
    
    # 对所有关键点对同时进行操作
    landmarks = priors_xy + predictions * variances[0] * priors_wh
    
    # 展平回[num_priors, 10]
    landmarks = landmarks.reshape(-1, 10)
    
    return landmarks