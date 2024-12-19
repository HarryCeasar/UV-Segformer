import cv2
import numpy as np

def median_filter_denoising(image_path, kernel_size=3):
    """
    使用中值滤波对遥感影像进行去噪。
    
    参数:
        image_path (str): 输入图像路径。
        kernel_size (int): 滤波窗口大小，默认为3x3。
        
    返回:
        denoised_image (numpy.ndarray): 去噪后的图像。
    """
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # IMREAD_UNCHANGED确保读取所有通道，包括alpha通道等
    
    if image is None:
        print(f"无法加载图像: {image_path}")
        return None

    # 应用中值滤波
    denoised_image = cv2.medianBlur(image, kernel_size)

    return denoised_image

if __name__ == '__main__':
    # 输入遥感影像路径
    img_path = 'path_to_your_remote_sensing_image.tif'

    # 调用去噪函数
    clean_image = median_filter_denoising(img_path, kernel_size=5)  # 可以尝试调整kernel_size

    if clean_image is not None:
        # 显示结果
        cv2.imshow('Denoised Image', clean_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存去噪后的图像
        cv2.imwrite('cleaned_image.tif', clean_image)