import pyiqa
import cv2
import torch

def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def calc_psnr(img_ref, img_test, device):
    iqa_metric = pyiqa.create_metric('psnry', device=device) # higher better

    return iqa_metric(img_test, img_ref)

