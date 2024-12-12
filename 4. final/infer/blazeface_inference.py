import numpy as np
import cv2
from .base_inference import BaseONNXInference
from models.retainface.priors import generate_priors, decode_boxes, decode_landmarks, BLAZEFACE_CONFIG

class BlazeFaceInference(BaseONNXInference):
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.02,
        nms_threshold: float = 0.4,
        input_size: tuple = (640, 640),
        prior_config: dict = None
    ):
        super().__init__(
            model_path=model_path,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
        )
        self.input_size = input_size
        self.prior_config = prior_config if prior_config is not None else BLAZEFACE_CONFIG
        self.priorbox = generate_priors(image_size = input_size, config = self.prior_config)
        self.bgr_means = (104, 117, 123)
    
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
        # 加载并预处理图像
        original_image = cv2.imread(image_path)
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
        # 减去RGB均值
        normalized_image = after_padding_image - self.bgr_means
        normalized_image = normalized_image.transpose(2, 0, 1)  # HWC to CHW
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
        # 解包模型输出
        # 从ONNX模型输出中提取定位、置信度和关键点信息
        # outputs[0] - 边界框回归值(loc)
        # outputs[1] - 置信度分数(conf) 
        # outputs[2] - 人脸关键点坐标(landmarks)
        # squeeze(0)用于移除批次维度,因为是单张图片推理
        loc, conf, landmarks = ort_outputs[0].squeeze(0), ort_outputs[1].squeeze(0), ort_outputs[2].squeeze(0)
         # 生成先验框
        boxes = decode_boxes(
            loc, 
            self.priorbox, 
            self.prior_config['variance']
        )
        landmarks = decode_landmarks(
            landmarks, 
            self.priorbox, 
            self.prior_config['variance']
        )

        # 获取padding和缩放信息
        scale = image_info['scale']
        dw, dh = image_info['pad']

        # 减去padding并还原缩放
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] * self.input_size[1] - dw) / scale  # x坐标
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * self.input_size[0] - dh) / scale  # y坐标

        # 裁剪到原始图像范围内
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_info['width'])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_info['height'])
    
        # 获取置信度分数
        scores = conf[:, 1]
        
        # 根据置信度阈值过滤
        inds = scores > self.conf_threshold
        boxes, landmarks, scores = boxes[inds], landmarks[inds], scores[inds]
        
        order = scores.argsort()[::-1]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        # 应用NMS
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self._nms(detections, self.nms_threshold)
        detections, landmarks = detections[keep], landmarks[keep]

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
