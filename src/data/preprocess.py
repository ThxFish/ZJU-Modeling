import cv2
import os

def preprocess_image(image_path, output_path):
    """
    使用 OpenCV 读取并处理图像
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    # 示例: 转换为灰度或者进行大小缩放、直方图均衡化等
    # resized = cv2.resize(img, (640, 640))
    cv2.imwrite(output_path, img)

if __name__ == "__main__":
    pass
