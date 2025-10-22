import os
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import math
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import cv2

# 计算 PSNR（峰值信噪比）
def psnr(img1, img2):
    mse_val = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse_val == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse_val))


def ssim_val(img1, img2):
    """计算 SSIM"""
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 显式指定 data_range
    score, _ = ssim(img1_gray, img2_gray, full=True, data_range=255)
    return score



# =====================
# 图像对齐函数
# =====================
def align_image(ref_img, target_img):
    """自动配准并对齐 target_img 到 ref_img"""
    h, w = ref_img.shape[:2]

    # 调整尺寸一致
    target_img = cv2.resize(target_img, (w, h))

    # 灰度转换
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)

    try:
        cc, warp_matrix = cv2.findTransformECC(ref_gray, tgt_gray, warp_matrix, cv2.MOTION_AFFINE, criteria)
        aligned = cv2.warpAffine(target_img, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned
    except cv2.error:
        # ECC 可能失败则直接返回 resize 版本
        return target_img


# =====================
# 单方法评估函数
# =====================
def evaluate_enhanced_images(ground_truth_dir, enhanced_dir, device):
    psnr_list, ssim_list = [], []

    # 使用 tqdm 显示文件遍历进度
    for filename in tqdm(os.listdir(ground_truth_dir), desc="Evaluating", ncols=100):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        gt_path = os.path.join(ground_truth_dir, filename)
        enh_path = os.path.join(enhanced_dir, filename)

        if not os.path.exists(enh_path):
            print(f"⚠️ 跳过 {filename}：增强图不存在")
            continue

        gt_img = cv2.imread(gt_path)
        enh_img = cv2.imread(enh_path)
        if gt_img is None or enh_img is None:
            print(f"⚠️ 跳过 {filename}：图像读取失败")
            continue

        # === 自动对齐 ===
        aligned_enh = align_image(gt_img, enh_img)

        # === 计算指标 ===
        # 将图像从 numpy 数组转换为 tensor，并将它们移动到 GPU
        gt_tensor = torch.from_numpy(gt_img).float().to(device)
        aligned_enh_tensor = torch.from_numpy(aligned_enh).float().to(device)

        # === 计算指标 ===
        psnr_score = psnr(gt_tensor.cpu().numpy(), aligned_enh_tensor.cpu().numpy())
        ssim_score = ssim_val(gt_tensor.cpu().numpy(), aligned_enh_tensor.cpu().numpy())

        psnr_list.append(psnr_score)
        ssim_list.append(ssim_score)

    if psnr_list and ssim_list:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        print(f"✅ 平均 PSNR: {avg_psnr:.2f} dB | 平均 SSIM: {avg_ssim:.4f}")
    else:
        print("⚠️ 没有成功评估的图像。")


# =====================
# 批量评估函数
# =====================
def evaluate_all_methods(device):
    ground_truth_dir = "./dataset/UIEB/reference-890"
    methods = [
        "cnn_enhanced_images",
        "contrast_enhanced",
        "deblurred",
        "gamma_corrected",
        "histogram_equalized",
        "sharpened"
    ]

    print("开始评估所有增强方法...\n")

    # 使用 tqdm 包装方法遍历
    for method in tqdm(methods, desc="Evaluating methods", ncols=100):
        enhanced_dir = os.path.join("./optimal", method)
        print(f"🔹 正在评估方法：{method}")
        evaluate_enhanced_images(ground_truth_dir, enhanced_dir, device)
        print("-" * 60)


if __name__ == '__main__':
    # 选择设备（GPU 0）
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 执行评估
    evaluate_all_methods(device)
