import numpy as np
import cv2
import matplotlib.pyplot as plt

EPSILON = 1e-8


def esif_1d(a, b, c, d, k: float = 1.0) -> float:
    """
    ESIF 1D đúng theo paper:
        x = μ b + (1-μ) c
        μ = (k (c-d)^2 + 1) / (k ((a-b)^2 + (c-d)^2) + 2)

    a,b,c,d: 4 mẫu liên tiếp; x nằm giữa b và c.
    """
    a = float(a)
    b = float(b)
    c = float(c)
    d = float(d)

    diff1 = (a - b) ** 2
    diff2 = (c - d) ** 2

    numerator = k * diff2 + 1.0
    denominator = k * (diff1 + diff2) + 2.0

    mu = numerator / (denominator + EPSILON)
    # theo lý thuyết μ luôn trong (0,1), clip cho chắc
    mu = np.clip(mu, 0.0, 1.0)

    return mu * b + (1.0 - mu) * c


def esif_upscale_2x(image: np.ndarray, k: float = 1.0) -> np.ndarray:
    """
    ESIF 2D (2x) bám sát mô tả trong paper.

    Bước:
      1) Đặt ảnh LR vào vị trí (2r,2c) trên canvas HR.
      2) Nội suy ngang các pixel (2r, 2c+1) bằng ESIF 1D trên hàng.
      3) Nội suy dọc các pixel (2r+1, 2c) bằng ESIF 1D trên cột.
      4) Pixel tâm (2r+1, 2c+1):
         - ESIF theo hướng ngang, dùng các giá trị đã nội suy dọc (y-type)
         - ESIF theo hướng dọc, dùng các giá trị đã nội suy ngang (x-type)
         - z = (z_horiz + z_vert) / 2
    """
    if image.ndim != 2:
        raise ValueError("Hàm này hiện chỉ hỗ trợ ảnh xám (2D).")

    img = image.astype(np.float64)
    H, W = img.shape

    H2, W2 = 2 * H, 2 * W
    out = np.zeros((H2, W2), dtype=np.float64)

    # 0) Gán điểm gốc (decimated samples) vào vị trí chẵn-chẵn
    out[0::2, 0::2] = img

    # 1) Nội suy ngang: (2r, 2c+1)
    for r in range(H):
        row = 2 * r  # hàng chẵn trên canvas
        for c in range(W - 1):
            # b, c_ là 2 pixel LR gốc; a,d là hàng xóm xa hơn, clamp biên
            b_col = 2 * c
            c_col = 2 * c + 2
            a_col = max(2 * c - 2, 0)
            d_col = min(2 * c + 4, W2 - 2)

            a = out[row, a_col]
            b = out[row, b_col]
            c_ = out[row, c_col]
            d = out[row, d_col]

            out[row, 2 * c + 1] = esif_1d(a, b, c_, d, k=k)

        # xử lý pixel cuối hàng: lặp lại từ pixel trước
        out[row, W2 - 1] = out[row, W2 - 2]

    # 2) Nội suy dọc: (2r+1, 2c)
    for c in range(W):
        col = 2 * c  # cột chẵn
        for r in range(H - 1):
            b_row = 2 * r
            c_row = 2 * r + 2
            a_row = max(2 * r - 2, 0)
            d_row = min(2 * r + 4, H2 - 2)

            a = out[a_row, col]
            b = out[b_row, col]
            c_ = out[c_row, col]
            d = out[d_row, col]

            out[2 * r + 1, col] = esif_1d(a, b, c_, d, k=k)

        # xử lý pixel cuối cột: lặp lại từ pixel phía trên
        out[H2 - 1, col] = out[H2 - 2, col]

    # 3) Pixel tâm (2r+1, 2c+1) = z
    #    z được tính là trung bình của 2 nội suy 1D:
    #      - ngang dùng các y-type (đã nội suy dọc, ở hàng lẻ cột chẵn)
    #      - dọc dùng các x-type (đã nội suy ngang, ở hàng chẵn cột lẻ)
    for r in range(H - 1):
        for c in range(W - 1):
            center_row = 2 * r + 1
            center_col = 2 * c + 1

            # --- nội suy ngang quanh z: dùng hàng lẻ (y-type) ---
            # mẫu dọc hàng: ... a, b, z, c, d ...
            # b và c nằm ở cột chẵn, đã được nội suy dọc ở bước 2
            b_col = 2 * c
            c_col = 2 * c + 2
            a_col = max(2 * c - 2, 0)
            d_col = min(2 * c + 4, W2 - 2)

            a_h = out[center_row, a_col]
            b_h = out[center_row, b_col]
            c_h = out[center_row, c_col]
            d_h = out[center_row, d_col]
            z_horiz = esif_1d(a_h, b_h, c_h, d_h, k=k)

            # --- nội suy dọc quanh z: dùng cột lẻ (x-type) ---
            # b và c nằm ở hàng chẵn, đã được nội suy ngang ở bước 1
            b_row = 2 * r
            c_row = 2 * r + 2
            a_row = max(2 * r - 2, 0)
            d_row = min(2 * r + 4, H2 - 2)

            a_v = out[a_row, center_col]
            b_v = out[b_row, center_col]
            c_v = out[c_row, center_col]
            d_v = out[d_row, center_col]
            z_vert = esif_1d(a_v, b_v, c_v, d_v, k=k)

            out[center_row, center_col] = 0.5 * (z_horiz + z_vert)

    # 4) Xử lý biên còn lại cho các ô tâm ở hàng/cột cuối:
    #    copy từ láng giềng gần nhất (giống cách bạn làm trước)
    # hàng lẻ cuối cùng: copy từ hàng trên
    out[H2 - 1, 1::2] = out[H2 - 2, 1::2]
    # cột lẻ cuối cùng: copy từ cột trước
    out[1::2, W2 - 1] = out[1::2, W2 - 2]
    # góc cuối: copy từ góc gần nhất
    out[H2 - 1, W2 - 1] = out[H2 - 2, W2 - 2]

    return np.clip(out, 0, 255).astype(np.uint8)


def esif_upscale_color_2x(bgr_img: np.ndarray, k: float = 1.0) -> np.ndarray:
    b, g, r = cv2.split(bgr_img)
    b_up = esif_upscale_2x(b, k)
    g_up = esif_upscale_2x(g, k)
    r_up = esif_upscale_2x(r, k)
    return cv2.merge([b_up, g_up, r_up])


if __name__ == "__main__":
    # Ví dụ test nhanh với Lenna xám
    img = cv2.imread("test_images/lenna.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit("Không đọc được ảnh test_images/lenna.png")

    up_k1 = esif_upscale_2x(img, k=1.0)
    up_k3 = esif_upscale_2x(img, k=3.0)

    print("Kích thước gốc :", img.shape)
    print("Kích thước ESIF:", up_k1.shape)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("ESIF k=1")
    plt.imshow(up_k1, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("ESIF k=3")
    plt.imshow(up_k3, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
