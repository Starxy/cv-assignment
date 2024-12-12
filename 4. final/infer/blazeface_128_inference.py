import numpy as np
import cv2
from .base_inference import BaseONNXInference

class BlazeFace128Inference(BaseONNXInference):
    def __init__(
        self,
        model_path,
        anchors_path,
        conf_threshold=0.75,
        nms_threshold=0.3,
    ):
        super().__init__(
            model_path=model_path,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
        )

        self.input_size = (128, 128)
        self.anchors = np.load(anchors_path).astype(np.float32)
        
        self.score_clipping_thresh = 100.0
        
        # 坐标解码用的缩放参数
        self.x_scale = 128.0
        self.y_scale = 128.0
        self.h_scale = 128.0
        self.w_scale = 128.0
    
    def preprocess(self, image_path: str):
        """
        图像预处理
        Args:
            image_path: 图片路径
        Returns:
            预处理后的用于推理的数据 (NCHW)
            图片信息
        """
        # 加载并预处理图像
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = original_image.shape[:2]

        # 计算缩放比例
        # 为了保持图像宽高比,取高度和宽度缩放比例的较小值
        # 例如原图640x480,目标尺寸640x640,则r=1.0
        r = min(self.input_size[0] / orig_h, self.input_size[1] / orig_w)

        # 计算padding
        # new_unpad_size 是缩放后的实际尺寸,例如640x480->640x480
        new_unpad_size = (int(round(orig_w * r)), int(round(orig_h * r)))

        # resize图像
        # 如果缩放比例不为1,则需要resize
        # 当r<1时使用INTER_AREA插值(缩小图像),r>1时使用INTER_LINEAR插值(放大图像)
        resized_image = original_image
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            resized_image = cv2.resize(original_image, new_unpad_size, interpolation=interp)

        # letterbox padding
        # dw,dh是需要填充的像素数,例如宽度方向填充(640-480)/2=80个像素
        dw, dh = (self.input_size[1] - new_unpad_size[0]) // 2, (self.input_size[0] - new_unpad_size[1]) // 2
        # 计算上下左右需要填充的像素数
        # 例如640x480的图像,上下各填充80像素变成640x640
        top, bottom = dh, self.input_size[0] - new_unpad_size[1] - dh
        left, right = dw, self.input_size[1] - new_unpad_size[0] - dw
        # 使用固定的灰色值(114,114,114)填充边框
        after_padding_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        after_padding_image = after_padding_image.transpose(2, 0, 1)  # HWC to CHW
        normalized_image = after_padding_image / 127.5 - 1.0

        # Add batch dimension (1, C, H, W)
        image_info = {
            "height": original_image.shape[0],
            "width": original_image.shape[1],
            "path": image_path,
            "scale": r,
            "pad": (dw, dh)  # 记录padding信息用于后处理
        }
        
        return np.expand_dims(normalized_image, axis=0).astype(np.float32), image_info
    
    def postprocess(self, ort_outputs, image_info):
        """
        后处理输出结果
        Args:
            ort_outputs: 模型输出
            image_info: 图片原始信息
        Returns:
            检测结果列表，每个元素是 Nx5 的数组:
              [ymin,xmin,ymax,xmax] 表示边界框坐标 使用原始尺寸的绝对坐标
              [score] 表示置信分数
        """
        raw_boxes, raw_scores = ort_outputs
        # 去掉batch维度,因为只处理单张图片
        raw_boxes = raw_boxes.squeeze(0)
        raw_scores = raw_scores.squeeze(0)
        # 解码预测框
        detection_boxes = self._decode_boxes(raw_boxes)  # 直接传入raw_boxes,不需要添加batch维度
        
        # sigmoid处理分数
        raw_scores = np.clip(raw_scores, -self.score_clipping_thresh, self.score_clipping_thresh)
        detection_scores = 1 / (1 + np.exp(-raw_scores))
        detection_scores = detection_scores.squeeze(axis=-1)
        
        # 应用分数阈值过滤
        mask = detection_scores >= self.conf_threshold
        boxes = detection_boxes[mask]
        scores = detection_scores[mask][:, np.newaxis]
        
        # 将边界框和分数拼接
        detections = np.concatenate((boxes, scores), axis=-1)
        
        # 应用加权非极大值抑制
        faces = self._weighted_non_max_suppression(detections)
        
        # 只保留边界框坐标和置信度分数
        if len(faces) > 0:
            faces = np.array(faces)
            faces = faces[:, [0,1,2,3,16]]  # 只取ymin,xmin,ymax,xmax,score
            
            # 减去padding并转换为原始图像的绝对坐标
            dw, dh = image_info['pad']
            input_size = self.input_size[0]  # 128
            
            # 先将归一化坐标转换为网络输入尺寸的绝对坐标
            faces[:, [0,2]] *= input_size  # ymin, ymax
            faces[:, [1,3]] *= input_size  # xmin, xmax
            
            # 减去padding
            faces[:, [0,2]] -= dh  # ymin, ymax
            faces[:, [1,3]] -= dw  # xmin, xmax
            
            # 缩放回原始图像尺寸
            faces[:, [0,2]] /= image_info['scale']  # ymin, ymax
            faces[:, [1,3]] /= image_info['scale']  # xmin, xmax
            
            # 裁剪到原始图像范围内
            faces[:, [0,2]] = np.clip(faces[:, [0,2]], 0, image_info['height'])
            faces[:, [1,3]] = np.clip(faces[:, [1,3]], 0, image_info['width'])
            
        else:
            faces = np.zeros((0, 5))  # 5表示[ymin,xmin,ymax,xmax,score]
            
        return faces

    def _decode_boxes(self, raw_boxes):
        """将模型的原始预测框转换为实际坐标
        
        Args:
            raw_boxes: 模型输出的原始预测框数据,shape为[num_anchors, 16]
                      其中16=4(边界框)+12(6个关键点坐标)
                      
        Returns:
            boxes: 解码后的预测框数据,shape与输入相同
                  包含:
                  - 边界框坐标(ymin,xmin,ymax,xmax)
                  - 6个关键点坐标(x1,y1,...,x6,y6)
        """
        # 创建与输入相同shape的零数组用于存储结果
        boxes = np.zeros_like(raw_boxes)
        
        # 解码边界框中心点坐标
        # 公式: center = offset/scale * anchor_size + anchor_center 
        x_center = raw_boxes[..., 0] / self.x_scale * self.anchors[:, 2] + self.anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale * self.anchors[:, 3] + self.anchors[:, 1]
        
        # 解码边界框宽高
        # 公式: size = offset/scale * anchor_size
        w = raw_boxes[..., 2] / self.w_scale * self.anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * self.anchors[:, 3]
        
        # 将中心点+宽高格式转换为左上右下角点格式
        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax
        
        # 解码6个人脸关键点坐标
        # 每个关键点包含(x,y)两个坐标,从索引4开始
        for k in range(6):
            offset = 4 + k*2
            # 使用与边界框中心点相同的解码方式
            keypoint_x = raw_boxes[..., offset] / self.x_scale * self.anchors[:, 2] + self.anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / self.y_scale * self.anchors[:, 3] + self.anchors[:, 1]
            boxes[..., offset] = keypoint_x
            boxes[..., offset + 1] = keypoint_y
            
        return boxes
    
    def _calculate_iou(self, box, boxes):
        """计算IOU"""
        # 计算交集
        xmin = np.maximum(box[1], boxes[:, 1])
        ymin = np.maximum(box[0], boxes[:, 0])
        xmax = np.minimum(box[3], boxes[:, 3])
        ymax = np.minimum(box[2], boxes[:, 2])
        
        intersection = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        
        # 计算并集
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection
        
        return intersection / union
    
    def _weighted_non_max_suppression(self, detections):
        """加权非极大值抑制"""
        if len(detections) == 0:
            return []
            
        output_detections = []
        
        # 按置信度分数降序排序
        remaining = np.argsort(detections[:, 16])[::-1]
        
        while len(remaining) > 0:
            detection = detections[remaining[0]]
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            
            # 计算IOU
            ious = self._calculate_iou(first_box, other_boxes)
            
            # 找出重叠的检测框
            mask = ious > self.nms_threshold
            overlapping = remaining[mask]
            remaining = remaining[~mask]
            
            # 加权平均重叠的检测框
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :16]
                scores = detections[overlapping, 16:17]
                total_score = np.sum(scores)
                weighted = np.sum(coordinates * scores, axis=0) / total_score
                detection = np.concatenate([weighted, [total_score / len(overlapping)]])
                
            output_detections.append(detection)
            
        return output_detections