import cv2
import numpy as np
import esif

img = cv2.imread("test_images/s1.png", cv2.IMREAD_COLOR)  # BGR, uint8
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

k = 1.0  # có thể thử 0.5, 1, 2 ...
up = esif.esif_upscale_2x_gray(img_rgb, k=k)

# nếu muốn lưu lại kiểu uint8:
up_uint8 = np.clip(up, 0, 255).astype(np.uint8)
up_bgr = cv2.cvtColor(up_uint8, cv2.COLOR_RGB2BGR)
cv2.imwrite("output_esif.png", up_bgr)
