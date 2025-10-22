import os
from PIL import Image, ImageFilter
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

# 锐化图像
def sharpen_image(image):
    return image.filter(ImageFilter.SHARPEN)

# 保存增强后的图像
def save_sharpened_images(raw_images, filenames):
    save_dir = './optimal/sharpened'
    os.makedirs(save_dir, exist_ok=True)
    
    # 使用tqdm显示进度条
    for img, filename in tqdm(zip(raw_images, filenames), total=len(raw_images), desc="锐化处理", ncols=100):
        file_name_without_extension = filename.split('.png')[0]
        # 保存文件时使用原文件名中的编号作为标识
        enhanced_filename = f'{file_name_without_extension}.png'
        sharpened_img = sharpen_image(img)
        sharpened_img.save(f'{save_dir}/{enhanced_filename}')

if __name__ == '__main__':
    # 加载raw-890数据集中的图像
    raw_images, filenames = load_images('./dataset/UIEB/raw-890')
    # 对图像进行锐化处理并保存
    save_sharpened_images(raw_images, filenames)
