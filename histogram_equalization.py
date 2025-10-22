import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm  # 导入tqdm库

# 加载原始图像
def load_images(directory):
    images = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.png'):  # 仅加载PNG格式的图像
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            images.append(img)
            filenames.append(filename)  # 保存文件名
    return images, filenames

# 彩色图像的直方图均衡化
def color_histogram_equalization(image):
    img_array = np.array(image)  # 将PIL图像转换为NumPy数组
    if len(img_array.shape) == 3:  # 如果是彩色图像（3个通道）
        r, g, b = cv2.split(img_array)  # 分离RGB三个通道
        r_eq = cv2.equalizeHist(r)  # 对红色通道进行均衡化
        g_eq = cv2.equalizeHist(g)  # 对绿色通道进行均衡化
        b_eq = cv2.equalizeHist(b)  # 对蓝色通道进行均衡化
        eq_img_array = cv2.merge([r_eq, g_eq, b_eq])  # 合并均衡化后的RGB通道
    else:
        eq_img_array = cv2.equalizeHist(img_array)  # 如果是灰度图像，直接进行均衡化

    return Image.fromarray(eq_img_array)  # 转回PIL图像

# 保存增强后的图像
def save_histogram_equalized_images(raw_images, filenames):
    save_dir = './optimal/histogram_equalized'
    os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在
    
    # 使用tqdm显示进度条
    for img, filename in tqdm(zip(raw_images, filenames), total=len(raw_images), desc="直方图均衡化", ncols=100):
        file_name_without_extension = filename.split('.png')[0]  # 去除文件扩展名
        # 为文件命名时使用原文件名中的编号
        enhanced_filename = f'{file_name_without_extension}_enhanced.png'
        eq_img = color_histogram_equalization(img)  # 进行直方图均衡化
        eq_img.save(f'{save_dir}/{enhanced_filename}')  # 保存增强后的图像

if __name__ == '__main__':
    # 加载raw-890数据集中的图像
    raw_images, filenames = load_images('./dataset/UIEB/raw-890')
    # 对图像进行直方图均衡化并保存
    save_histogram_equalized_images(raw_images, filenames)
