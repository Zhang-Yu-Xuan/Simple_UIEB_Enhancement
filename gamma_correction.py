import os
import numpy as np
from PIL import Image
from tqdm import tqdm  # 导入tqdm库

# 加载原始图像
def load_images(directory):
    images = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            images.append(img)
            filenames.append(filename)  # 保存文件名
    return images, filenames

# 伽马校正
def gamma_correction(image, gamma=1.5):
    img_array = np.array(image) / 255.0  # 归一化到[0, 1]
    img_array = np.power(img_array, gamma)  # 应用伽马变换
    img_array = np.uint8(np.clip(img_array * 255, 0, 255))  # 恢复到[0, 255]
    return Image.fromarray(img_array)

# 保存增强后的图像
def save_gamma_corrected_images(raw_images, filenames):
    save_dir = './optimal/gamma_corrected'
    os.makedirs(save_dir, exist_ok=True)
    
    # 使用tqdm显示进度条
    for img, filename in tqdm(zip(raw_images, filenames), total=len(raw_images), desc="伽马校正", ncols=100):
        file_name_without_extension = filename.split('.png')[0]
        # 保存文件时使用原文件名中的编号作为标识
        enhanced_filename = f'{file_name_without_extension}.png'
        gamma_img = gamma_correction(img)
        gamma_img.save(f'{save_dir}/{enhanced_filename}')

if __name__ == '__main__':
    # 加载raw-890数据集中的图像
    raw_images, filenames = load_images('./dataset/UIEB/raw-890')
    # 对图像进行伽马校正并保存
    save_gamma_corrected_images(raw_images, filenames)
