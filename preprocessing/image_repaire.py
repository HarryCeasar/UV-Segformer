import cv2
import numpy as np

def inpaint_image(image_path, mask_path):
    # 读取图像和掩码
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)  # 以灰度模式读取

    # 检查图像和掩码是否成功加载
    if image is None:
        print(f"无法加载图像: {image_path}")
        return None
    if mask is None:
        print(f"无法加载掩码: {mask_path}")
        return None

    # 使用Telea's算法进行图像修补
    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    return result

if __name__ == '__main__':
    # 输入图像路径和对应的掩码路径
    img_path = 'path_to_your_image.jpg'
    mask_path = 'path_to_your_mask.png'

    # 调用修补函数
    repaired_image = inpaint_image(img_path, mask_path)

    if repaired_image is not None:
        # 显示结果
        cv2.imshow('Repaired Image', repaired_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存修补后的图像
        cv2.imwrite('repaired_image.jpg', repaired_image)
