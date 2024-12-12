import numpy as np
import cv2
from .base_inference import BaseONNXInference
from models.retainface.priors import generate_priors, decode_boxes, decode_landmarks, RETINAFACE_CONFIG

class RetinaFaceInference(BaseONNXInference):
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.02,
        nms_threshold: float = 0.4,
        pre_nms_topk: int = 5000,
        post_nms_topk: int = 750,
        prior_config: dict = None
    ):
        """
        初始化RetinaFace ONNX推理类
        Args:
            model_path: ONNX模型路径
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
            pre_nms_topk: 预NMS处理时保留的候选框数量
            post_nms_topk: 后处理时保留的候选框数量
            prior_config: 先验框配置，如果为None则使用默认配置
        """
        super().__init__(
            model_path=model_path,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
        )
        self.bgr_means = (104, 117, 123)
        self.prior_config = prior_config if prior_config is not None else RETINAFACE_CONFIG
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
    
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
        
        image_info = {
            "height": original_image.shape[0],
            "width": original_image.shape[1],
            "path": image_path
        }

        image = np.float32(original_image)
        image -= self.bgr_means
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = np.expand_dims(image, axis=0)  # Add batch dimension (1, C, H, W)
        return image, image_info
    
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
        # 解包模型输出
        # 从ONNX模型输出中提取定位、置信度和关键点信息
        # outputs[0] - 边界框回归值(loc)
        # outputs[1] - 置信度分数(conf) 
        # outputs[2] - 人脸关键点坐标(landmarks)
        # squeeze(0)用于移除批次维度,因为是单张图片推理
        loc, conf, landmarks = ort_outputs[0].squeeze(0), ort_outputs[1].squeeze(0), ort_outputs[2].squeeze(0)
         # 生成先验框
        priorbox = generate_priors(image_size = (image_info['height'], image_info['width']), config = self.prior_config)
        boxes = decode_boxes(
            loc, 
            priorbox, 
            self.prior_config['variance']
        )
        landmarks = decode_landmarks(
            landmarks, 
            priorbox, 
            self.prior_config['variance']
        )

        # 调整边界框和关键点的尺度
        bbox_scale = np.array([image_info['width'], image_info['height']] * 2)
        boxes = boxes * bbox_scale
    
        # 获取置信度分数
        scores = conf[:, 1]
        
        # 根据置信度阈值过滤
        inds = scores > self.conf_threshold
        boxes, landmarks, scores = boxes[inds], landmarks[inds], scores[inds]
        
        # 按分数排序并保留前K个
        order = scores.argsort()[::-1][:self.pre_nms_topk]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        # 应用NMS
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self._nms(detections, self.nms_threshold)
        detections, landmarks = detections[keep], landmarks[keep]

        # 保留最终的TopK个检测结果
        detections, landmarks = detections[:self.post_nms_topk], landmarks[:self.post_nms_topk]
        result = np.concatenate([
                    detections[:, 1:2],  # ymin
                    detections[:, 0:1],  # xmin 
                    detections[:, 3:4],  # ymax
                    detections[:, 2:3],  # xmax
                    detections[:, 4:5]   # score
                ], axis=1)
        return result
    
    def _nms(self, dets, threshold):
        """
        应用非极大值抑制(NMS)来减少基于阈值的重叠边界框。

        参数:
            dets: 检测结果数组,每行格式为 [x1, y1, x2, y2, score]
            threshold: 用于抑制的 IoU 阈值

        返回:
            抑制后保留的边界框索引列表
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]

        return keep
