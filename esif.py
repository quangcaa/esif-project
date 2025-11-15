import numpy as np


# p - nonlinear weighting
def _interp_weight(a, b, c, d, k):
    sL2 = (a - b) * (a - b)
    sR2 = (c - d) * (c - d)

    num = k * sL2 + 1.0
    den = k * (sL2 + sR2) + 2.0

    # avoid division 0
    den = np.where(den == 0, 1e-12, den)

    p = num / den
    return p


def _get_clamped(img, i, j):
    """
    Lấy giá trị img[i, j] với i, j được clamp vào [0, H-1], [0, W-1].
    Hỗ trợ cả 2D lẫn 3D (ảnh màu).
    """
    H, W = img.shape[:2]
    i = np.clip(i, 0, H - 1)
    j = np.clip(j, 0, W - 1)
    return img[i, j]


def esif_upscale_2x_gray(src, k=1.0):
    """    ESIF cho ảnh xám (2D) phóng to 2x (H,W) -> (2H-1, 2W-1).

    src: np.ndarray 2D
    k  : tham số phi tuyến (k=0 -> nội suy tuyến tính)
    """
    if src.ndim != 2:
        raise ValueError("esif_upscale_2x_gray chỉ nhận ảnh 2D (grayscale).")

    src = src.astype(np.float64)
    H, W = src.shape
    H2 = 2 * H - 1
    W2 = 2 * W - 1

    dst = np.zeros((H2, W2), dtype=np.float64)

    # 1) Gán các pixel gốc vào vị trí (2i, 2j)
    for i in range(H):
        for j in range(W):
            dst[2 * i, 2 * j] = src[i, j]

    # 2) Nội suy điểm nằm giữa hai pixel theo chiều ngang (loại x)
    #    dst[2*i, 2*j+1] giữa src[i,j] (b) và src[i,j+1] (c)
    for i in range(H):
        for j in range(W - 1):
            # 4 pixel liên tiếp theo hàng: a,b,c,d
            a = _get_clamped(src, i, j - 1)
            b = _get_clamped(src, i, j)
            c = _get_clamped(src, i, j + 1)
            d = _get_clamped(src, i, j + 2)

            if k == 0:
                # nội suy tuyến tính
                x = 0.5 * (b + c)
            else:
                p = _interp_weight(a, b, c, d, k)
                x = p * b + (1.0 - p) * c

            dst[2 * i, 2 * j + 1] = x

    # 3) Nội suy điểm nằm giữa hai pixel theo chiều dọc (loại y)
    #    dst[2*i+1, 2*j] giữa src[i,j] (b) và src[i+1,j] (c)
    for i in range(H - 1):
        for j in range(W):
            a = _get_clamped(src, i - 1, j)
            b = _get_clamped(src, i, j)
            c = _get_clamped(src, i + 1, j)
            d = _get_clamped(src, i + 2, j)

            if k == 0:
                y = 0.5 * (b + c)
            else:
                p = _interp_weight(a, b, c, d, k)
                y = p * b + (1.0 - p) * c

            dst[2 * i + 1, 2 * j] = y

    # 4) Nội suy điểm nằm giữa 4 pixel (loại z) dùng 2 mặt nạ 1D:
    #    - một theo hàng (trung tâm giữa 2 điểm đã nội suy dọc)
    #    - một theo cột (trung tâm giữa 2 điểm đã nội suy ngang)
    #    rồi lấy trung bình 2 kết quả, đúng ý “mean of the two filters”
    for i in range(H - 1):
        for j in range(W - 1):
            ci = 2 * i + 1
            cj = 2 * j + 1

            # --- Mặt nạ ngang quanh z ---
            # ta coi z nằm giữa b_h = dst[ci, 2*j] và c_h = dst[ci, 2*j+2]
            a_h = _get_clamped(dst, ci, 2 * j - 2)  # xa trái
            b_h = _get_clamped(dst, ci, 2 * j)  # trái gần (điểm y đã nội suy)
            c_h = _get_clamped(dst, ci, 2 * j + 2)  # phải gần (điểm y đã nội suy)
            d_h = _get_clamped(dst, ci, 2 * j + 4)  # xa phải

            if k == 0:
                z_h = 0.5 * (b_h + c_h)
            else:
                p_h = _interp_weight(a_h, b_h, c_h, d_h, k)
                z_h = p_h * b_h + (1.0 - p_h) * c_h

            # --- Mặt nạ dọc quanh z ---
            # z nằm giữa b_v = dst[2*i, cj] và c_v = dst[2*i+2, cj]
            a_v = _get_clamped(dst, 2 * i - 2, cj)  # trên xa
            b_v = _get_clamped(dst, 2 * i, cj)  # trên gần (điểm x đã nội suy)
            c_v = _get_clamped(dst, 2 * i + 2, cj)  # dưới gần (điểm x đã nội suy)
            d_v = _get_clamped(dst, 2 * i + 4, cj)  # dưới xa

            if k == 0:
                z_v = 0.5 * (b_v + c_v)
            else:
                p_v = _interp_weight(a_v, b_v, c_v, d_v, k)
                z_v = p_v * b_v + (1.0 - p_v) * c_v

            # --- Giá trị cuối cùng là trung bình ---
            dst[ci, cj] = 0.5 * (z_h + z_v)

    return dst


def esif_upscale_2x(src, k=1.0):
    """
    ESIF cho ảnh xám hoặc ảnh màu (H,W) hoặc (H,W,3).
    Trả về ảnh float64; nếu muốn trả về uint8 thì tự clip và ép kiểu.

    src: np.ndarray 2D hoặc 3D
    k  : tham số phi tuyến (k=0 -> tuyến tính)
    """
    src = np.asarray(src)

    if src.ndim == 2:
        return esif_upscale_2x_gray(src, k=k)
    elif src.ndim == 3:
        # áp dụng từng kênh độc lập
        H, W, C = src.shape
        H2 = 2 * H - 1
        W2 = 2 * W - 1
        dst = np.zeros((H2, W2, C), dtype=np.float64)
        for c in range(C):
            dst[:, :, c] = esif_upscale_2x_gray(src[:, :, c], k=k)
        return dst
    else:
        raise ValueError("Ảnh đầu vào phải là 2D (xám) hoặc 3D (màu).")
