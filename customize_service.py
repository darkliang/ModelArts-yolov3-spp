import os
import log
from model_service.pytorch_model_service import PTServingBaseService
from models import Darknet
import torch.nn as nn
import torch
import numpy as np
import cv2
import torchvision
from pathlib import Path
import glob
import math
from PIL import Image


logger = log.getLogger(__name__)


class PTVisionService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        global CLASS_INDEX
        # 调用父类的初始化方法，方法参数为（model_name, model_path）
        super(PTVisionService, self).__init__(model_name, model_path)
        # self.model定义为用户load后的模型
        # self.model_path为pth，pt模型的本地全路径，可以使用dirname方法提取模型的目录路径，
        # 根据目录路径加载模型包内的分类标签文件，imagenet_class_index.json和pth，pt文件在同一个目录下
        dir_path = os.path.dirname(os.path.realpath(self.model_path))
        self.model = get_darknet(model_path, os.path.join(
            dir_path, "cfg/yolov3-spp.cfg"))
        CLASS_INDEX = load_classes(os.path.join(dir_path, "data/coco.names"))
        import sys
        print(sys.version)

    def _preprocess(self, data):
        # 预处理成{key:input_batch_var}，input_batch_var为模型输入张量
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                with Image.open(file_content) as pil_image:
                    img0 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                img = letterbox(img0, new_shape=(608, 352))[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
                img = np.ascontiguousarray(
                    img, dtype=np.float32)  # uint8 to fp16/fp32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if torch.cuda.is_available():
                    img = torch.from_numpy(img).cuda()
                else:
                    img = torch.from_numpy(img)
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                preprocessed_data[k] = [img, img0.shape]

        return preprocessed_data

    def _inference(self, data):
        result = {}
        for k, v in data.items():
            det = non_max_suppression(self.model(v[0])[0], 0.3, 0.5)[0]
            if det is not None:
                det[:, :4] = scale_coords(
                    v[0].shape[2:], det[:, :4], v[1]).round()
            result[k] = det
        return result

    def _postprocess(self, data):
        # 根据标签索引到图片的分类结果
        for k, v in data.items():
            results = []
            if v is not None:
                for *xyxy, conf, _, cls in v:
                    result = {"box": [int(xyxy[1]), int(xyxy[3]), int(xyxy[0]), int(xyxy[2])],
                              "confidence": conf.item(),
                              "class": CLASS_INDEX[int(cls)]
                              }
                    results.append(result)
            return results


__all__ = ['yolov3-spp', 'yolov3-spp']


def get_darknet(model_path, cfg_path):
    model = Darknet(cfg_path, (608, 352))

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.load_state_dict(torch.load(
            model_path, map_location="cuda:0")['model'])
        model.to(device)
        print('use gpu')
    else:
        device = torch.device('cpu')
        model.load_state_dict(torch.load(
            model_path, map_location=device)['model'])
        print('use cpu')

    model = model.eval()

    return model


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    # filter removes empty strings (such as last line)
    return list(filter(None, names))


def letterbox(img, new_shape=(416, 416), color=(128, 128, 128),
              auto=True, scaleFill=False, scaleup=True, interp=cv2.INTER_AREA):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = max(new_shape) / max(shape)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        # INTER_AREA is better, INTER_LINEAR is faster
        img = cv2.resize(img, new_unpad, interpolation=interp)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(
        min=0, max=img_shape[1])  # clip x
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(
        min=0, max=img_shape[0])  # clip y


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    iou = inter_area / union_area  # iou
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union_area) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + \
                ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.5):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_conf, class)
    """

    # (pixels) minimum and maximium box width and height
    min_wh, max_wh = 2, 30000

    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        # Experiment: Prior class size rejection
        # x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        # a = w * h  # area
        # ar = w / (h + 1e-16)  # aspect ratio
        # n = len(w)
        # log_w, log_h, log_a, log_ar = torch.log(w), torch.log(h), torch.log(a), torch.log(ar)
        # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
        # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
        # from scipy.stats import multivariate_normal
        # for c in range(60):
        # shape_likelihood[:, c] =
        #   multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])

        # Multiply conf by class conf to get combined confidence
        class_conf, class_pred = pred[:, 5:].max(1)
        pred[:, 4] *= class_conf

        # # Merge classes (optional)
        # class_pred[(class_pred.view(-1,1) == torch.LongTensor([2, 3, 5, 6, 7]).view(1,-1)).any(1)] = 2
        #
        # # Remove classes (optional)
        # pred[class_pred != 2, 4] = 0.0

        # Select only suitable predictions
        i = (pred[:, 4] > conf_thres) & (pred[:, 2:4] > min_wh).all(1) & (pred[:, 2:4] < max_wh).all(1) & \
            torch.isfinite(pred).all(1)
        pred = pred[i]

        # If none are remaining => process next image
        if len(pred) == 0:
            continue

        # Select predicted classes
        class_conf = class_conf[i]
        class_pred = class_pred[i].unsqueeze(1).float()

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
        pred = torch.cat((pred[:, :5], class_conf.unsqueeze(1), class_pred), 1)

        # Get detections sorted by decreasing confidence scores
        pred = pred[(-pred[:, 4]).argsort()]

        # Set NMS method https://github.com/ultralytics/yolov3/issues/679
        # 'OR', 'AND', 'MERGE', 'VISION', 'VISION_BATCHED'
        # MERGE is highest mAP, VISION is fastest

        # Non-maximum suppression
        det_max = []
        for c in pred[:, -1].unique():
            dc = pred[pred[:, -1] == c]  # select class c
            n = len(dc)
            if n == 1:
                det_max.append(dc)  # No NMS required if only 1 prediction
                continue
            elif n > 500:
                # limit to first 500 boxes: https://github.com/ultralytics/yolov3/issues/117
                dc = dc[:500]

            while len(dc) > 1:
                iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                if iou.max() > 0.5:
                    det_max.append(dc[:1])
                dc = dc[1:][iou < nms_thres]  # remove ious > threshold

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort

    return output
