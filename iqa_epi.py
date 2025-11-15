import numpy as np
import cv2


def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def calc_epi(img_ref, img_test, edge_thresh=10.0, ksize=3):
    """
    Calculate Edge Preservation Index (EPI)

    - img_ref: GT
    - img_test: ảnh sau nội suy
    - edge_thresh: ngưỡng gradient để chọn pixel là biên
    - ksize: kích thước kernel Sobel (thường 3)

    Return:
    - epi: value [0, 1] | higher better
    """

    # check size
    assert img_ref.shape[:2] == img_test.shape[:2], "Hai ảnh phải cùng size"

    # return to grayscale and float
    ref = to_gray(img_ref).astype(np.float32)
    test = to_gray(img_test).astype(np.float32)

    # Gradient của ảnh gốc
    gx_ref = cv2.Sobel(ref, cv2.CV_32F, 1, 0, ksize=ksize)
    gy_ref = cv2.Sobel(ref, cv2.CV_32F, 0, 1, ksize=ksize)
    mag_ref = np.sqrt(gx_ref ** 2 + gy_ref ** 2)

    # Gradient của ảnh nội suy
    gx_test = cv2.Sobel(test, cv2.CV_32F, 1, 0, ksize=ksize)
    gy_test = cv2.Sobel(test, cv2.CV_32F, 0, 1, ksize=ksize)
    mag_test = np.sqrt(gx_test ** 2 + gy_test ** 2)

    # Chọn các pixel biên trên ảnh gốc
    edge_mask = mag_ref > edge_thresh
    if not np.any(edge_mask):
        # Không có biên nào đủ mạnh
        return 0.0

    g1 = mag_ref[edge_mask].ravel()
    g2 = mag_test[edge_mask].ravel()

    # Tính cosine similarity giữa vector gradient magnitude
    num = np.sum(g1 * g2)
    den = np.sqrt(np.sum(g1 ** 2) * np.sum(g2 ** 2))

    if den == 0:
        return 0.0

    epi = num / den

    # Đảm bảo trong [0,1] (do lỗi số có thể hơi lệch)
    epi = float(np.clip(epi, 0.0, 1.0))
    return epi
