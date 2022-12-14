import os.path as osp
import sys

import cv2
import numpy as np
import torchvision
import albumentations as A
import torch

here = osp.dirname(osp.abspath(__file__))


class SNCD(object):
    def __init__(self, **kwargs):
        super().__init__()
        self.code_root = kwargs.pop('code_root', here)
        self.setup()

        model = build_model(
            choice=kwargs.pop('choice', 'cdp_UnetPlusPlus'),
            encoder_name=kwargs.pop('encoder_name', "timm-efficientnet-b2"),
            encoder_weights=kwargs.pop('encoder_weights', "noisy-student"),
            decoder_attention_type=kwargs.pop('decoder_attention_type', None),
            in_channels=kwargs.pop('in_channels', 3),
            classes=kwargs.pop('classes', 2),
            siam_encoder=kwargs.pop('siam_encoder', True),
            fusion_form=kwargs.pop('fusion_form', 'concat'),
        )

        weights = kwargs.pop('weights', None)
        self.load_checkpoint(weights, {"state_dict": model})

        self.model = model
        self.scale = kwargs.pop('scale', False)
        self.device = torch.device(kwargs.pop('device', 'cpu'))
        self.names = kwargs.pop('names', ['changed'])

    def setup(self):
        sys.path.insert(0, self.code_root)
        # sys.path.insert(0, "/Users/shinian/proj/code/ChangeDetection")

        from build import build_model
        global build_model

    def load_checkpoint(self, path, ret_state):
        if not osp.exists(path): return ret_state
        # import pdb;pdb.set_trace()
        ckpt = torch.load(path, map_location='cpu')
        for item, state in ckpt.items():
            if item in ret_state:
                new_state = {k.replace('module.',''):v for k,v in state.items() \
                        # if 'sam12.conv3.' not in kk\
                        # and 'sam23.conv3.' not in kk
                        } if item=='state_dict' else state
                ret_state[item].load_state_dict(new_state, strict=False)
                print(f"loaded {item} in {path}.")
            else:
                print(f"extra {item} in {path}.")
        return ret_state

    def convert_pred_to_labelme(self, pred):
        # import pdb;pdb.set_trace()
        pred = torch.nn.functional.interpolate(pred.float(), self.img0.shape[:2])
        shapes = []
        pred = pred.cpu().numpy()[0][0]
        contours, hierarchy = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # area = cv2.contourArea(contour)
            rect = cv2.minAreaRect(contour)  
            box = np.int0(cv2.boxPoints(rect)) 
            points = box.tolist()            
            shapes.append({
                        "label":
                        self.names[0],
                        "points": points,
                        "group_id": None,
                        "shape_type":
                        "polygon",
                        "flags": {}
                    }) 
        return shapes

    def convert_img_to_tensor(self, img):
        # if isinstance(img, str): img = cv2.imread(img, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))[...,::-1] # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]
        return img

    def preprocess(self, x):

        img, img_1 = x

        if isinstance(img, str): 
            img = cv2.imread(img, -1)
        if isinstance(img_1, str): 
            img_1 = cv2.imread(img_1, -1)

        self.img0 = img

        img = self.convert_img_to_tensor(img)
        img_1 = self.convert_img_to_tensor(img_1)
        
        return img, img_1

    @torch.no_grad()
    def process(self, x):
        img, img_1 = x
        pred = self.model(img, img_1)
        return pred

    def postprocess(self, x):
        x = x.argmax(dim=1, keepdim=True)
        x = self.convert_pred_to_labelme(x)
        return x

    def __call__(self, x):
        x = self.preprocess(x)
        x = self.process(x)
        x = self.postprocess(x)
        return x

def sn(**kwargs):
    model = SNCD(**kwargs)
    return model

if __name__ == "__main__":
    import cv2
    import yaml

    with open('../plugin.yaml', 'r') as f:
        config = yaml.safe_load(f)
    img = cv2.imread('/Users/shinian/proj/data/stb/train/A/1147.tif', -1)
    img_1 = cv2.imread('/Users/shinian/proj/data/stb/train/B/1147.tif', -1)
    model = sn(**config['plugin']['plugins'][1]['args']['model_args'])
    x = model([img, img_1])
    print(x)