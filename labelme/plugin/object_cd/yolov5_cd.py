import os.path as osp
import sys

import cv2
import numpy as np
import torch

here = osp.dirname(osp.abspath(__file__))


class YOLOv5CD(object):
    def __init__(self, **kwargs):
        super().__init__()
        self.code_root = kwargs.pop('code_root', here)
        self.setup()

        conf_thres = kwargs.pop('conf_thres', 0.3)
        iou_thres = kwargs.pop('iou_thres', 0.3)
        prob_thres = kwargs.pop('prob_thres', 0.3)

        imgsz = kwargs.pop('imgsz', 640)
        weights = kwargs.pop('weights', './best.pt')
        device = kwargs.pop('device', 'cpu')
        stride = kwargs.pop('stride', 32)
        names = kwargs.pop('names', ['changed'])
        augment = kwargs.pop('augment', False)

        device = select_device(device)
        half = device.type != 'cpu'
        model = DetectMultiBackend(weights, device=device, dnn=False)
        stride, pt, jit, onnx, engine = model.stride, model.pt, model.jit, model.onnx, model.engine

        if isinstance(imgsz, int): imgsz = [imgsz, imgsz]
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        half &= (pt or jit or onnx or engine) and device.type != 'cpu'
        if pt or jit:
            model.model.half() if half else model.model.float()
        model.warmup(imgsz=(1, 2, 3, *imgsz), half=half)  # warmup
        model.eval()

        self.model = model
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.prob_thres = prob_thres
        self.names = names
        self.imgsz = imgsz
        self.stride = stride
        self.device = device
        self.augment = augment
        self.half = half

        self.img0 = None

    def __call__(self, x):
        x = self.preprocess(x)
        x = self.process(x)
        x = self.postprocess(x)
        return x

    def setup(self):
        sys.path.insert(0, self.code_root)
        # sys.path.insert(0, "/Users/shinian/proj/code/yolov5_cd/yolov5")
        from models.common import DetectMultiBackend
        from utils.augmentations import letterbox
        from utils.general import (check_img_size,
                                   non_max_suppression,
                                   scale_coords)
        from utils.torch_utils import select_device

        global DetectMultiBackend, check_img_size, non_max_suppression,\
             scale_coords, select_device, letterbox

    def convert_img_to_tensor(self, img):
        # if isinstance(img, str): img = cv2.imread(img, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))[...,::-1] # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]
        return img

    def convert_pred_to_labelme(self, pred):
        shapes = []
        for i, det in enumerate(pred):  # detections per image
            # gn = torch.tensor(self.img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                # import pdb;pdb.set_trace()
                det[:, :4] = scale_coords(self.img.shape[2:], det[:, :4],
                                          self.img0.shape).round()

                for *xyxy, conf, cls in det:
                    if conf < self.prob_thres:
                        continue
                    x1, y1, x2, y2 = map(float, xyxy)
                    x1 = self.img0.shape[0] - x1
                    x2 = self.img0.shape[0] - x2
                    shapes.append({
                        "label":
                        self.names[int(cls)],
                        "points": [[x1, y1], [x1, y2], [x2, y2], [x2, y1]],
                        "group_id": None,
                        "shape_type":
                        "polygon",
                        "flags": {}
                    })

        # return json.dumps(shapes, indent=4)
        return shapes

    def preprocess(self, x):
        imgsz = self.imgsz
        stride = self.stride

        img, img_1 = x
        if isinstance(img, str): 
            img = cv2.imread(img, -1)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if isinstance(img_1, str): 
            img_1 = cv2.imread(img_1, -1)
            # img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)

        self.img0 = img
        img, img_1, ratio, pad = letterbox(img,
                                           img_1,
                                           new_shape=imgsz,
                                           stride=stride,
                                           auto=True)
        img = self.convert_img_to_tensor(img)
        img_1 = self.convert_img_to_tensor(img_1)
        x = torch.stack([img, img_1], dim=1)
        self.img = img

        return x

    @torch.no_grad()
    def process(self, x):
        pred = self.model(x, augment=self.augment, visualize=False)
        return pred

    def postprocess(self, x):
        x = non_max_suppression(
            x,
            self.conf_thres,
            self.iou_thres,
            agnostic=False,
            classes=None,
        )
        x = self.convert_pred_to_labelme(x)
        return x


def yolov5(**kwargs):
    model = YOLOv5CD(**kwargs)
    return model


if __name__ == "__main__":
    import cv2
    import yaml

    with open('../plugin.yaml', 'r') as f:
        config = yaml.safe_load(f)
    img = cv2.imread('/Users/shinian/proj/data/stb/train/A/1147.tif', -1)
    img_1 = cv2.imread('/Users/shinian/proj/data/stb/train/B/1147.tif', -1)
    model = YOLOv5CD(**config['plugin']['plugins'][0]['args']['model_args'])
    x = model([img, img_1])
    print(x)
