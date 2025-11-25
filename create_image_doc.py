import cv2
import numpy as np
import os
from glob import glob
import os.path as osp


def create_fixed_grid_with_labels_grayscale(input_folder, output_path, grid_shape=(2, 4), cell_size=(200, 200),
                                            margin=10, font_scale=0.5, font_thickness=1):
    """
    Tạo khung grid grayscale với mỗi ô cố định, ảnh giữ tỉ lệ, căn giữa, và ghi tên file dưới ảnh.
    """
    # Lấy ảnh
    image_files = sorted(glob(os.path.join(input_folder, "*.*")))
    num_needed = grid_shape[0] * grid_shape[1]
    images = []
    labels = []

    for f in image_files[:num_needed]:
        img = cv2.imread(f)
        if img is not None:
            # Chuyển sang grayscale và convert lại 3 kênh để dễ hiển thị
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            images.append(img_gray)
            labels.append(osp.splitext(osp.basename(f))[0])  # tên file không có .png/.jpg

    # Nếu không đủ ảnh, điền ô trắng
    while len(images) < num_needed:
        images.append(np.ones((cell_size[1], cell_size[0], 3), dtype=np.uint8) * 255)
        labels.append("")

    rows, cols = grid_shape
    cell_w, cell_h = cell_size
    total_width = cols * cell_w + (cols - 1) * margin
    total_height = rows * cell_h + (rows - 1) * margin + 20 * rows  # thêm khoảng trống cho text
    grid_image = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255  # canvas trắng

    # Chọn font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Đặt ảnh vào từng ô
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols

        # Resize ảnh giữ tỉ lệ
        h, w = img.shape[:2]
        scale = min(cell_w / w, (cell_h - 20) / h)  # trừ 20px cho text
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        # Tọa độ đặt ảnh (căn giữa ô)
        x_start = c * (cell_w + margin) + (cell_w - new_w) // 2
        y_start = r * (cell_h + margin) + (cell_h - new_h - 20) // 2  # 20px cho text

        grid_image[y_start:y_start + new_h, x_start:x_start + new_w] = resized

        # Viết tên file dưới ảnh
        label = labels[idx]
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x = c * (cell_w + margin) + (cell_w - text_size[0]) // 2
        text_y = y_start + new_h + 15  # 15px từ ảnh xuống text
        cv2.putText(grid_image, label, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    # Hiển thị và lưu
    cv2.imshow("Grayscale Grid with Labels", grid_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(output_path, grid_image)
    print(f"Saved grayscale grid with labels to {output_path}")


# Ví dụ dùng:
create_fixed_grid_with_labels_grayscale("test_images/qualitative-eval", "output/grid_gray.jpg", grid_shape=(2, 4), cell_size=(200, 200),
                                        margin=10)
