import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import pyiqa
import torch
import nedi
import esif
import iqa_epi
import iqa_psnr


# p - nonlinear weighting
def interpolation_weight(a, b, c, d, k: float):
    sL2 = (a - b) * (a - b)
    sR2 = (c - d) * (c - d)
    tu = k * sR2 + 1
    mau = k * (sL2 + sR2) + 2
    return tu / mau


# border processing
def gather(img, i, j):
    i = np.clip(i, 0, img.shape[0] - 1)
    j = np.clip(j, 0, img.shape[1] - 1)
    return img[i, j]


# x2 res for grayscale img with esif
def _upsample2x_gray(grayscale_img: np.ndarray, k: float) -> np.ndarray:
    H, W = grayscale_img.shape
    out = np.zeros((H * 2, W * 2), dtype=grayscale_img.dtype)

    # sao chép điểm gốc vào (2i, 2j)
    out[0::2, 0::2] = grayscale_img

    # các điểm giữa theo chiều ngang: (2i, 2j+1)
    for i in range(H):
        for j in range(W):
            # a b x c d dọc ngang
            a = gather(grayscale_img, i, j - 1)
            b = gather(grayscale_img, i, j)
            c = gather(grayscale_img, i, j + 1)
            d = gather(grayscale_img, i, j + 2)
            p = interpolation_weight(a, b, c, d, k)
            x = p * b + (1 - p) * c
            out[2 * i, 2 * j + 1] = x

    # các điểm giữa theo chiều dọc: (2i+1, 2j)
    for i in range(H):
        for j in range(W):
            a = gather(grayscale_img, i - 1, j)
            b = gather(grayscale_img, i, j)
            c = gather(grayscale_img, i + 1, j)
            d = gather(grayscale_img, i + 2, j)
            p = interpolation_weight(a, b, c, d, k)
            y = p * b + (1 - p) * c
            out[2 * i + 1, 2 * j] = y

    # điểm trung tâm chéo: (2i+1, 2j+1)
    for i in range(H):
        for j in range(W):
            # Mặt nạ chéo chính: a b ? c d (b=(i,j), c=(i+1,j+1))
            a1 = gather(grayscale_img, i - 1, j - 1)
            b1 = gather(grayscale_img, i, j)
            c1 = gather(grayscale_img, i + 1, j + 1)
            d1 = gather(grayscale_img, i + 2, j + 2)
            p1 = interpolation_weight(a1, b1, c1, d1, k)
            z1 = p1 * b1 + (1 - p1) * c1

            # Mặt nạ chéo phụ: a b ? c d (b=(i+1,j), c=(i,j+1))
            a2 = gather(grayscale_img, i + 2, j - 1)
            b2 = gather(grayscale_img, i + 1, j)
            c2 = gather(grayscale_img, i, j + 1)
            d2 = gather(grayscale_img, i - 1, j + 2)
            p2 = interpolation_weight(a2, b2, c2, d2, k)
            z2 = p2 * b2 + (1 - p2) * c2

            out[2 * i + 1, 2 * j + 1] = 0.5 * (z1 + z2)

    return out


def _to_float(img: np.ndarray):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0, 255.0, np.uint8
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0, 65535.0, np.uint16
    # giả sử float [0,1]
    return img.astype(np.float32), 1.0, np.float32


def _from_float(imgf: np.ndarray, peak: float, dtype):
    if np.issubdtype(dtype, np.integer):
        return np.clip(np.round(imgf * peak), 0, peak).astype(dtype)
    return np.clip(imgf, 0.0, 1.0).astype(dtype)


def esif_2x(image: np.ndarray, k: float = 0.8) -> np.ndarray:
    """
    Nội suy 2x theo thuật toán nhạy biên.
    - Hỗ trợ ảnh xám và RGB/BGR (C ở chiều cuối).
    - k ~ [0,1.2]; cao hơn -> sắc hơn nhưng có thể gắt biên.
    """
    if image.ndim == 2:
        imgf, peak, dtype = _to_float(image)
        out = _upsample2x_gray(imgf, k)
        return _from_float(out, peak, dtype)

    if image.ndim == 3 and image.shape[2] in (3, 4):
        # Xử lý từng kênh để tránh sai sắc độ
        ch = image.shape[2]
        imgf, peak, dtype = _to_float(image)
        outs = []
        for c in range(ch):
            outs.append(_upsample2x_gray(imgf[:, :, c], k))
        out = np.stack(outs, axis=2)
        return _from_float(out, peak, dtype)

    raise ValueError("Ảnh phải là (H,W) hoặc (H,W,C) với C=3/4.")


def parse_rect(rect_str):
    # "x,y,w,h" -> 4 int
    try:
        x, y, w, h = map(int, rect_str.split(","))
        assert w > 0 and h > 0
        return x, y, w, h
    except Exception:
        raise ValueError("--crop phải có dạng x,y,w,h (số nguyên dương)")


def safe_crop(img, x, y, w, h):
    H, W = img.shape[:2]
    x0 = max(0, min(x, W - 1))
    y0 = max(0, min(y, H - 1))
    x1 = max(x0 + 1, min(x + w, W))
    y1 = max(y0 + 1, min(y + h, H))
    return img[y0:y1, x0:x1]


def main():
    parser = argparse.ArgumentParser(description="ESIF")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--k", type=float, default=0.8, help="Edge sensitivity")
    parser.add_argument("--crop", default=None, help="crop: x,y,w,h (coors in original image)")
    args = parser.parse_args()

    # check gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: {}".format(device))

    # read input image
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    img_ts = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float() / 255.0
    if img is None:
        raise FileNotFoundError("Can not find image: {}".format(args.input))
    h, w = img.shape
    print(img.shape)

    # gaussian + decimation
    # img_blur = cv2.GaussianBlur(img, (5, 5), sigmaX=1, sigmaY=1, borderType=cv2.BORDER_REFLECT)
    # img_lr = img_blur[::2, ::2]
    img_lr = img

    # bilinear +  interpolation + esif + nedi
    # format : BGR (H, W, 3)
    bilinear = cv2.resize(img_lr, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    bicubic = cv2.resize(img_lr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    esif_img = esif_2x(img_lr, k=args.k)
    esif_img_2 = esif.esif_upscale_2x_gray(img_lr, k=args.k)
    nedi_img = nedi.EDI_predict(img_lr, 4, 2)

    # calculate IQA
    # PSNR
    psnr1 = iqa_psnr.calc_psnr(img_ts, bilinear, device)
    psnr2 = iqa_psnr.calc_psnr(img_ts, bicubic, device)
    psnr3 = iqa_psnr.calc_psnr(img_ts, esif_img, device)
    print("PSNR || Bilinear: {} | Bicubic: {} | ESIF: {}".format(psnr1, psnr2, psnr3))

    # EPI
    epi1 = iqa_epi.calc_epi(bilinear, img)
    epi2 = iqa_epi.calc_epi(bicubic, img)
    epi3 = iqa_epi.calc_epi(esif_img, img)
    print("EPI || Bilinear: {} | Bicubic: {} | ESIF: {}".format(epi1, epi2, epi3))

    # GMSD - Gradient Magnitude Similarity Deviation
    # gmsd_metric = pyiqa.create_metric('gmsd', device=device)
    # print("Lower better : {}".format(iqa_metric.lower_better))
    # gmsd1 = gmsd_metric(bilinear_ts, img_ts)
    # gmsd2 = gmsd_metric(bicubic_ts, img_ts)
    # gmsd4 = gmsd_metric(esif_img_ts, img_ts)
    # print("GMSD\nBilinear: {}\nBicubic: {}\nLanczos: {}\nESIF: {}".format(gmsd1, gmsd2, gmsd3, gmsd4))

    # visualization
    fig, ax = plt.subplots(2, 3, figsize=(10, 6))
    ax[0, 0].imshow(img, cmap='gray')
    ax[0, 0].set_title("Original")
    ax[0, 1].imshow(bilinear, cmap='gray')
    ax[0, 1].set_title("Bilinear")
    ax[0, 2].imshow(bicubic, cmap='gray')
    ax[0, 2].set_title("Bicubic")
    ax[1, 0].imshow(esif_img_2, cmap='gray')
    ax[1, 0].set_title("ESIF_2")
    ax[1, 1].imshow(esif_img, cmap='gray')
    ax[1, 1].set_title("ESIF")
    ax[1, 2].imshow(nedi_img, cmap='gray')
    ax[1, 2].set_title("NEDI")

    for a in ax.ravel(): a.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
