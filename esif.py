import numpy as np
import cv2
import matplotlib.pyplot as plt


def interpolate_1d_nonlin(b, c, a=None, d=None, k=1.0):
    if a is None:
        a = b
    if d is None:
        d = c

    # μ = ((c-d)^2 + 1) / (k*((a-b)^2 + (c-d)^2) + 2)
    mu_num = (c - d) ** 2 + 1.0
    mu_den = k * ((a - b) ** 2 + (c - d) ** 2) + 2.0
    mu = mu_num / mu_den

    # x = μ b + (1 - μ) c
    x = mu * b + (1.0 - mu) * c
    return x


def upscale_2x_edge_preserving(image, k=1.0):
    """
    Phóng to ảnh 2× theo phương pháp phi tuyến bảo toàn biên.
    Các bước:
      1) Nội suy điểm giữa theo chiều ngang
      2) Nội suy điểm giữa theo chiều dọc
      3) Nội suy điểm chéo còn thiếu (diagonal refinement)
    """
    # Đảm bảo dạng float
    img = image.astype(np.float32)
    H, W = img.shape

    # Canvas 2×
    out = np.zeros((2 * H, 2 * W), dtype=np.float32)

    # Đặt điểm gốc (copy vào ô chẵn-chẵn)
    out[0::2, 0::2] = img

    # 1) Nội suy ngang: ô chẵn - lẻ (giữa hai gốc theo trục x)
    # vị trí (2r, 2c+1) nằm giữa (2r, 2c) và (2r, 2c+2)
    for r in range(H):
        for c in range(W - 1):
            b = out[2 * r, 2 * c]  # pixel trái
            c_ = out[2 * r, 2 * c + 2]  # pixel phải
            a = out[2 * r, max(2 * c - 2, 0)]  # trước b
            d = out[2 * r, min(2 * c + 4, 2 * W - 2)]  # sau c_
            out[2 * r, 2 * c + 1] = interpolate_1d_nonlin(b, c_, a, d, k=k)

        # Biên phải: dùng lặp lại (an toàn)
        out[2 * r, 2 * W - 1] = out[2 * r, 2 * W - 2]

    # 2) Nội suy dọc: ô lẻ - chẵn (giữa hai gốc theo trục y)
    # vị trí (2r+1, 2c) nằm giữa (2r, 2c) và (2r+2, 2c)
    for r in range(H - 1):
        for c in range(W):
            b = out[2 * r, 2 * c]  # pixel trên
            c_ = out[2 * r + 2, 2 * c]  # pixel dưới
            a = out[max(2 * r - 2, 0), 2 * c]  # trước b
            d = out[min(2 * r + 4, 2 * H - 2), 2 * c]  # sau c_
            out[2 * r + 1, 2 * c] = interpolate_1d_nonlin(b, c_, a, d, k=k)

        # Biên dưới: dùng lặp lại
        out[2 * H - 1, 2 * c] = out[2 * H - 2, 2 * c]

    # 3) Nội suy chéo: ô lẻ - lẻ (giữa 4 điểm đã có)
    # vị trí (2r+1, 2c+1) nằm giữa (2r,2c),(2r,2c+2),(2r+2,2c),(2r+2,2c+2)
    for r in range(H - 1):
        for c in range(W - 1):
            # Ta nội suy theo hai hướng và phối hợp:
            # - Ngang: giữa (2r,2c+1) và (2r,2c+3) nhưng 2c+3 có thể không tồn tại, dùng cặp quanh chéo
            # - Dọc: giữa (2r+1,2c) và (2r+3,2c)
            # Để ổn định, ta dùng 2 đường: ngang qua hàng trên và dưới, rồi trung bình phi tuyến.
            top_left = out[2 * r, 2 * c]
            top_right = out[2 * r, 2 * c + 2]
            bot_left = out[2 * r + 2, 2 * c]
            bot_right = out[2 * r + 2, 2 * c + 2]

            # Nội suy theo hàng trên (giữa top_left, top_right)
            a_t = out[2 * r, max(2 * c - 2, 0)]
            d_t = out[2 * r, min(2 * c + 4, 2 * W - 2)]
            x_top = interpolate_1d_nonlin(top_left, top_right, a_t, d_t, k=k)

            # Nội suy theo hàng dưới (giữa bot_left, bot_right)
            a_b = out[2 * r + 2, max(2 * c - 2, 0)]
            d_b = out[2 * r + 2, min(2 * c + 4, 2 * W - 2)]
            x_bot = interpolate_1d_nonlin(bot_left, bot_right, a_b, d_b, k=k)

            # Nội suy theo cột trái (giữa top_left, bot_left)
            a_l = out[max(2 * r - 2, 0), 2 * c]
            d_l = out[min(2 * r + 4, 2 * H - 2), 2 * c]
            x_left = interpolate_1d_nonlin(top_left, bot_left, a_l, d_l, k=k)

            # Nội suy theo cột phải (giữa top_right, bot_right)
            a_r = out[max(2 * r - 2, 0), 2 * c + 2]
            d_r = out[min(2 * r + 4, 2 * H - 2), 2 * c + 2]
            x_right = interpolate_1d_nonlin(top_right, bot_right, a_r, d_r, k=k)

            # Kết hợp bốn ước lượng (trung bình có trọng số nhẹ theo độ chênh)
            candidates = np.array([x_top, x_bot, x_left, x_right], dtype=np.float32)
            diffs = np.array([
                abs(top_left - top_right),
                abs(bot_left - bot_right),
                abs(top_left - bot_left),
                abs(top_right - bot_right)
            ], dtype=np.float32) + 1e-6
            weights = 1.0 / (diffs ** 0.5)  # trọng số ưu tiên hướng ít thay đổi hơn
            out[2 * r + 1, 2 * c + 1] = np.sum(candidates * weights) / np.sum(weights)

    # Xử lý biên còn lại (hàng/cột cuối lẻ): sao chép từ láng giềng gần nhất
    out[2 * H - 1, 1::2] = out[2 * H - 2, 1::2]
    out[1::2, 2 * W - 1] = out[1::2, 2 * W - 2]

    return out


if __name__ == "__main__":
    img = cv2.imread("test_images/Set5/butterfly.png", cv2.IMREAD_GRAYSCALE)  # BGR, uint8
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    up1 = upscale_2x_edge_preserving(img, k=1.0)
    up2 = upscale_2x_edge_preserving(img, k=3.0)

    print("Kích thước gốc:", img.shape)
    print("Kích thước phóng to:", up1.shape)

    # Lưu ảnh dưới dạng PNG
    # cv2.imwrite("up1.png", (up1 * 255).astype(np.uint8))
    # cv2.imwrite("up2.png", (up2 * 255).astype(np.uint8))

    plt.subplot(1, 3, 1)
    plt.title("Origin")
    plt.imshow(img, cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("k=1")
    plt.imshow(up1, cmap="gray")
    plt.subplot(1, 3, 3)
    plt.title("k=3")
    plt.imshow(up2, cmap="gray")

    plt.show()
