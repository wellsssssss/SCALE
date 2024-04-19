import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter, RandomResizedCrop
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, ConcatDataset
batch_size=32
# 加载数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # RandomHorizontalFlip(),  # 50%的概率水平翻转
    # RandomRotation(15),  # 在（-15, 15）范围内旋转
    # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

source_dataset = datasets.ImageFolder('/root/autodl-tmp/sepico/OfficeHome/Real World', transform=transform)


target_dataset = datasets.ImageFolder('/root/autodl-tmp/sepico/OfficeHome/Product', transform=transform)

source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=4)