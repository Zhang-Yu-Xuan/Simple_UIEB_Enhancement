import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from torch.cuda.amp import autocast, GradScaler


# ==============================
# 一、CNN模型（残差结构，自动修复灰图）
# ==============================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + residual)

class CNNModel(nn.Module):
    def __init__(self, num_res_blocks=4):
        super().__init__()
        self.head = nn.Conv2d(3, 64, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        self.tail = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        x = self.relu(self.head(x))
        x = self.res_blocks(x)
        x = self.tail(x)
        return torch.clamp(x, 0, 1)  # 保证输出在 [0,1]

# ==============================
# 二、自定义数据集
# ==============================
class CustomDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.png')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.filenames[idx])
        image = Image.open(img_name).convert('RGB')
        gt_name = img_name.replace('raw-890', 'reference-890')
        ground_truth = Image.open(gt_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
            ground_truth = self.transform(ground_truth)

        return image, ground_truth, self.filenames[idx]

# ==============================
# 三、动态padding collate
# ==============================
def collate_fn(batch):
    imgs, gts, filenames = zip(*batch)
    sizes = [(img.shape[1], img.shape[2]) for img in imgs]
    max_h = max(h for h, _ in sizes)
    max_w = max(w for _, w in sizes)

    padded_imgs, padded_gts = [], []
    for img, gt in zip(imgs, gts):
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        padded_imgs.append(F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0))
        padded_gts.append(F.pad(gt, (0, pad_w, 0, pad_h), mode='constant', value=0))

    return torch.stack(padded_imgs), torch.stack(padded_gts), filenames, sizes

# ==============================
# 四、感知损失（增强细节，使用本地 VGG16 权重）
# ==============================
class PerceptualLoss(nn.Module):
    def __init__(self, device, vgg_weights_path='./vgg16-397923af.pth'):
        super().__init__()
        vgg = models.vgg16(weights=None).features.to(device).eval()
        state_dict_full = torch.load(vgg_weights_path, map_location=device)
        state_dict_features = {k.replace('features.', ''): v for k, v in state_dict_full.items() if 'features.' in k}
        vgg.load_state_dict(state_dict_features)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.device = device

    def forward(self, output, target):
        # output, target 形状: [B, 3, H, W]，直接传入 VGG
        # 注意 VGG 输入要求归一化到 [0,1] 或 [-1,1]，如果训练输入是 [0,1] 可以直接用
        return F.l1_loss(self.vgg(output), self.vgg(target))



# ==============================
# 五、训练函数
# ==============================
def train(model, loader, criterion_pixel, criterion_perceptual, optimizer, device, num_epochs=1):
    model.train()
    scaler = GradScaler()  # 创建梯度缩放器
    for epoch in range(num_epochs):
        running_loss = 0
        for imgs, gts, _, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100):
            imgs, gts = imgs.to(device), gts.to(device)
            optimizer.zero_grad()
            
            with autocast():  # 自动混合精度上下文
                out = model(imgs)
                loss_pixel = criterion_pixel(out, gts)
                loss_perc = criterion_perceptual(out, gts)
                loss = loss_pixel + 0.1 * loss_perc
            
            scaler.scale(loss).backward()  # 缩放梯度
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(loader):.4f}")
        
# ==============================
# 六、保存/加载模型
# ==============================
def save_model(model, path='./models/cnn_model.pth'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model

# ==============================
# 七、推理与保存
# ==============================
def infer_and_save(model, loader, device, save_dir='./optimal/cnn_enhanced_images'):
    model.eval()  # 确保模型在评估模式
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for imgs, _, filenames, sizes in tqdm(loader, desc="Enhancing", ncols=100):
            imgs = imgs.to(device)
            
            # 自动混合精度
            with torch.cuda.amp.autocast():
                out = model(imgs)
            
            # 保存结果
            for i, filename in enumerate(filenames):
                h, w = sizes[i]
                img_out = out[i, :, :h, :w].cpu()
                img_out = transforms.ToPILImage()(img_out)
                name_no_ext = filename.split('.png')[0]
                img_out.save(os.path.join(save_dir, f'{name_no_ext}.png'))

# ==============================
# 八、主函数
# ==============================
if __name__ == '__main__':
    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    print(f"使用 GPU 数量: {n_gpu}")

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomDataset('./dataset/UIEB/raw-890', transform)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    model = CNNModel().to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)

    criterion_pixel = nn.L1Loss()
    criterion_perc = PerceptualLoss(device, vgg_weights_path='./vgg16-397923af.pth')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练
    # train(model, train_loader, criterion_pixel, criterion_perc, optimizer, device, num_epochs=10)
    # save_model(model)

    # 推理
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 单卡推理
    model = CNNModel()
    model = load_model(model, './models/cnn_model.pth', device)  # 单卡加载
    model.eval()  # 不再使用 DataParallel

    dataset_raw = CustomDataset('./dataset/UIEB/raw-890', transform)
    loader_raw = DataLoader(dataset_raw, batch_size=8, shuffle=False, collate_fn=collate_fn)

    infer_and_save(model, loader_raw, device)
   