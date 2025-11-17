import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import pyiqa
import torch
import esif
import nedi
import iqa_epi
import iqa_psnr
import iqa_gmsd
import os
import shutil


def main():
    parser = argparse.ArgumentParser(description="ESIF")
    parser.add_argument("input", help="input image path")
    # parser.add_argument("output", help="output image path")
    parser.add_argument("--k", type=float, default=0.8, help="edge sensitivity")
    parser.add_argument("--crop", default=None, help="crop: x,y,w,h")
    args = parser.parse_args()

    # ===== SET UP DEVICE =====
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    # ===== INPUT =====
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Can not find image: {args.input}")
    h, w = img.shape
    print(img.shape)

    # ===== PREPROCESS (GAUSSIAN + DECIMATION) =====
    img_blur = cv2.GaussianBlur(img, (5, 5), sigmaX=1, sigmaY=1, borderType=cv2.BORDER_REFLECT)
    img_lr = img[::2, ::2]

    # ===== INTERPOLATION =====
    nearest = cv2.resize(img_lr, (w, h), interpolation=cv2.INTER_NEAREST)
    bilinear = cv2.resize(img_lr, (w, h), interpolation=cv2.INTER_LINEAR)
    bicubic = cv2.resize(img_lr, (w, h), interpolation=cv2.INTER_CUBIC)
    esif_img = esif.upscale_2x_edge_preserving(img_lr, k=args.k)
    nedi_img = nedi.EDI_predict(img_lr, 4, 2)

    # ===== SAVE RESULT IMAGES =====
    save_dir = "results"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    cv2.imwrite(os.path.join(save_dir, "nearest.png"), nearest)
    cv2.imwrite(os.path.join(save_dir, "bilinear.png"), bilinear)
    cv2.imwrite(os.path.join(save_dir, "bicubic.png"), bicubic)
    cv2.imwrite(os.path.join(save_dir, "esif.png"), esif_img)
    cv2.imwrite(os.path.join(save_dir, "nedi.png"), nedi_img)

    print(f"Saved images to folder: {save_dir}")

    # ===== IQA EVAL =====
    # === PSNR ===
    psnr1 = iqa_psnr.calc_psnr(args.input, "results/nearest.png", device)
    psnr2 = iqa_psnr.calc_psnr(args.input, "results/bilinear.png", device)
    psnr3 = iqa_psnr.calc_psnr(args.input, "results/bicubic.png", device)
    psnr4 = iqa_psnr.calc_psnr(args.input, "results/esif.png", device)
    psnr5 = iqa_psnr.calc_psnr(args.input, "results/nedi.png", device)
    print(
        f"===== PSNR⬆️ =====\nNearest: {psnr1.item()}\nBilinear: {psnr2.item()}\nBicubic: {psnr3.item()}\nESIF: {psnr4.item()}\nNEDI: {psnr5.item()}")

    # === EPI ===
    epi1 = iqa_epi.calc_epi(nearest, img)
    epi2 = iqa_epi.calc_epi(bilinear, img)
    epi3 = iqa_epi.calc_epi(bicubic, img)
    epi4 = iqa_epi.calc_epi(esif_img, img)
    epi5 = iqa_epi.calc_epi(nedi_img, img)
    print(
        f"===== EPI⬆️ =====\nNearest: {epi1}\nBilinear: {epi2}\nBicubic: {epi3}\nESIF: {epi4}\nNEDI: {epi5}")

    # === GMSD ===
    gmsd1 = iqa_gmsd.calc_gmsd(args.input, "results/nearest.png", device)
    gmsd2 = iqa_gmsd.calc_gmsd(args.input, "results/bilinear.png", device)
    gmsd3 = iqa_gmsd.calc_gmsd(args.input, "results/bicubic.png", device)
    gmsd4 = iqa_gmsd.calc_gmsd(args.input, "results/esif.png", device)
    gmsd5 = iqa_gmsd.calc_gmsd(args.input, "results/nedi.png", device)
    print(
        f"===== GMSD⬇️  =====\nNearest: {gmsd1.item()}\nBilinear: {gmsd2.item()}\nBicubic: {gmsd3.item()}\nESIF: {gmsd4.item()}\nNEDI: {gmsd5.item()}")

    # ===== VISUALIZATION =====
    fig, ax = plt.subplots(2, 3, figsize=(10, 6))
    ax[0, 0].imshow(img, cmap='gray')
    ax[0, 0].set_title("Origin")
    ax[0, 1].imshow(nearest, cmap='gray')
    ax[0, 1].set_title("Nearest")
    ax[0, 2].imshow(bicubic, cmap='gray')
    ax[0, 2].set_title("Bicubic")
    ax[1, 0].imshow(bicubic, cmap='gray')
    ax[1, 0].set_title("Bicubic")
    ax[1, 1].imshow(esif_img, cmap='gray')
    ax[1, 1].set_title("ESIF")
    ax[1, 2].imshow(nedi_img, cmap='gray')
    ax[1, 2].set_title("NEDI")

    for a in ax.ravel(): a.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
