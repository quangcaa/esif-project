import pyiqa
import cv2
import torch

def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def calc_psnr(img_ref, img_test, device):
    # create metric
    iqa_metric = pyiqa.create_metric('psnry', device=device)
    print("Lower better : {}".format(iqa_metric.lower_better))

    # convert to tensor (grayscale)
    img_tensor = torch.from_numpy(img_test).unsqueeze(0).unsqueeze(0).float() / 255.0

    return iqa_metric(img_tensor, img_ref)

