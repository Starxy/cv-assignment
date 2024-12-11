import numpy as np
import cv2
from .base_inference import BaseONNXInference

class BlazeFaceInference(BaseONNXInference):
    def __init__(
        self,
        model_path,
        anchors_path,
        conf_threshold=0.75,
        nms_threshold=0.3,
        input_size=(128, 128)
    ):
        super().__init__(
            model_path=model_path,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            input_size=input_size
        )
        self.anchors = np.load(anchors_path).astype(np.float32)
        
        # BlazeFace特定参数
        self.num_anchors = 896
        self.num_classes = 1
        self.num_coords = 16
        self.score_clipping_thresh = 100.0
        
        # 坐标解码用的缩放参数
        self.x_scale = 128.0
        self.y_scale = 128.0
        self.h_scale = 128.0
        self.w_scale = 128.0
    
    def preprocess(self, images):
        """
        图像预处理
        Args:
            images: 单张图片或图片列表，图片为使用 cv2 读入，格式为 HWC
        Returns:
            预处理后的批量数据(NCHW)，像素范围[-1,1]
        """
        # 将单张图片转换为批量格式
        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
            
        # 一次性处理整个批次
        # 1. 缩放
        resized = np.array([cv2.resize(img, self.input_size) for img in images])
        # 2. 归一化到[-1,1]
        normalized = resized / 127.5 - 1.0
        # 3. HWC -> NCHW 
        preprocessed = normalized.transpose(0, 3, 1, 2)
        
        # ONNX模型的输入一般要求float32类型,这里确保输入数据类型正确
        return preprocessed.astype(np.float32)
    
    def _decode_boxes(self, raw_boxes):
        """将预测框解码为实际坐标"""
        boxes = np.zeros_like(raw_boxes)
        
        # 解码中心点坐标
        x_center = raw_boxes[..., 0] / self.x_scale * self.anchors[:, 2] + self.anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale * self.anchors[:, 3] + self.anchors[:, 1]
        
        # 解码宽高
        w = raw_boxes[..., 2] / self.w_scale * self.anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * self.anchors[:, 3]
        
        # 转换为左上右下格式
        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax
        
        # 解码关键点
        for k in range(6):
            offset = 4 + k*2
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
    
    def postprocess(self, ort_outputs, image_info):
        """
        后处理输出结果
        Args:
            ort_outputs: [raw_boxes, raw_scores] 模型输出
            image_info: 图像信息
        Returns:
            检测结果列表，每个元素是 Nx5 的数组:
              [ymin,xmin,ymax,xmax] 表示边界框坐标 (相对坐标,范围0-1)
              [score] 表示置信分数
        """
        raw_boxes, raw_scores = ort_outputs
        
        # 解码预测框
        detection_boxes = self._decode_boxes(raw_boxes)
        
        # sigmoid处理分数
        raw_scores = np.clip(raw_scores, -self.score_clipping_thresh, self.score_clipping_thresh)
        detection_scores = 1 / (1 + np.exp(-raw_scores))
        detection_scores = detection_scores.squeeze(axis=-1)
        
        # 应用分数阈值
        mask = detection_scores >= self.conf_threshold
        
        # 处理每个批次的图像
        output_detections = []
        for i in range(raw_boxes.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            scores = detection_scores[i, mask[i]]
            
            # 合并框和分数
            detections = np.concatenate((boxes, scores[:, np.newaxis]), axis=-1)
            
            # 应用 加权非极大值抑制
            filtered_detections = self._weighted_non_max_suppression(detections)
            # 如果有检测结果，将其堆叠为数组并只保留边界框和置信度
            if filtered_detections:
                filtered_detections = np.stack(filtered_detections)
                # 只保留边界框坐标(前4个)和置信度(最后1个)
                filtered_detections = np.concatenate([
                    filtered_detections[:, :4],  # 边界框坐标
                    filtered_detections[:, -1:] # 置信度
                ], axis=1)
            else:
                filtered_detections = np.zeros((0, 5))  # 5 = 4(边界框) + 1(置信度)
            output_detections.append(filtered_detections)
            
        return output_detections if len(output_detections) > 1 else output_detections[0]
