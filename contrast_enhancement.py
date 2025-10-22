import os
from PIL import Image, ImageEnhance
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

# 对比度增强
def enhance_contrast(image, factor=2):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)  # factor为对比度增强因子

# 保存增强后的图像
def save_contrast_enhanced_images(raw_images, filenames):
    save_dir = './optimal/contrast_enhanced'
    os.makedirs(save_dir, exist_ok=True)
    
    # 使用tqdm显示进度条
    for img, filename in tqdm(zip(raw_images, filenames), total=len(raw_images), desc="对比度增强", ncols=100):
       
        file_name_without_extension = filename.split('.png')[0]
        # 保存文件时使用原文件名中的编号作为标识
        enhanced_filename = f'{file_name_without_extension}.png'
        enhanced_img = enhance_contrast(img)
        enhanced_img.save(f'{save_dir}/{enhanced_filename}')

if __name__ == '__main__':
    # 加载raw-890数据集中的图像
    raw_images, filenames = load_images('./dataset/UIEB/raw-890')
    # 对图像进行对比度增强并保存
    save_contrast_enhanced_images(raw_images, filenames)
