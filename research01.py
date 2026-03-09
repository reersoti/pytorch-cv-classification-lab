import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

# -----------------------------------------------------
#  STYLISH TERMINAL HEADER (как в research01)
# -----------------------------------------------------
def print_header(title):
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70 + "\n")

# -----------------------------------------------------
#  EMA (Exponential Moving Average)
# -----------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = \
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data = self.shadow[name]

# -----------------------------------------------------
# MixUp + CutMix helper
# -----------------------------------------------------
def mixup_cutmix(images, labels, alpha=1.0):
    if random.random() < 0.5:
        # MixUp
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(images.size(0)).to(images.device)
        mixed = lam * images + (1 - lam) * images[index, :]
        labels_a, labels_b = labels, labels[index]
        return mixed, labels_a, labels_b, lam, "mixup"
    else:
        # CutMix
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(images.size(0)).to(images.device)
        H, W = images.size(2), images.size(3)

        cut_w = int(W * math.sqrt(1 - lam))
        cut_h = int(H * math.sqrt(1 - lam))
        cx = random.randint(0, W)
        cy = random.randint(0, H)

        x1 = max(0, cx - cut_w // 2)
        x2 = min(W, cx + cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(H, cy + cut_h // 2)

        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        labels_a, labels_b = labels, labels[index]
        return mixed, labels_a, labels_b, lam, "cutmix"

# -----------------------------------------------------
# Label smoothing CE
# -----------------------------------------------------
class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_prob = nn.functional.log_softmax(pred, dim=1)
        with torch.no_grad():
            smoothed = torch.zeros_like(log_prob).fill_(self.smoothing / (n_classes - 1))
            smoothed.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        loss = (-smoothed * log_prob).sum(dim=1).mean()
        return loss

# -----------------------------------------------------
# ORIGINAL MODEL (не меняем архитектуру)
# -----------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class MyResNet50k(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.res1 = ResidualBlock(32, 64)
        self.res2 = ResidualBlock(64, 128)
        self.res3 = ResidualBlock(128, 128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 100)

    def forward(self, x):
        x = self.layer1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

# -----------------------------------------------------
# Data (оставим минимальный pipline)
# -----------------------------------------------------
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5071,0.4867,0.4408],
                         [0.2675,0.2565,0.2761]),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5071,0.4867,0.4408],
                         [0.2675,0.2565,0.2761]),
])

data_root = "./data"
train_set = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

# -----------------------------------------------------
# Model
# -----------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print_header("Loading model (MyResNet50k)")
model = MyResNet50k().to(device)
ema = EMA(model)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = LabelSmoothLoss(smoothing=0.1)

# Cosine LR with warmup
warmup_epochs = 3
total_epochs = 40

def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return float(epoch) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# -----------------------------------------------------
# Training
# -----------------------------------------------------
print_header("Start Training")
best_top1 = 0

for epoch in range(total_epochs):
    model.train()
    loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{total_epochs}]", ncols=100)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # MixUp / CutMix
        images, la, lb, lam, mode = mixup_cutmix(images, labels, alpha=1.0)

        optimizer.zero_grad()
        outputs = model(images)

        loss = lam * criterion(outputs, la) + (1 - lam) * criterion(outputs, lb)
        loss.backward()
        optimizer.step()

        ema.update(model)
        loop.set_postfix(loss=float(loss))

    scheduler.step()

    # ------------------ VAL ------------------
    model.eval()
    ema.apply_shadow(model)

    correct1, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = outputs.max(1)
            correct1 += pred.eq(labels).sum().item()
            total += labels.size(0)

    top1 = correct1 / total
    print(f"\nEpoch {epoch+1}:  Top-1 = {top1:.4f}   | LR = {scheduler.get_last_lr()[0]:.6f}")

    # Save best
    if top1 > best_top1:
        best_top1 = top1
        path = os.path.join(checkpoint_dir, f"best50k_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), path)
        print(f"Saved BEST model → {path}")

print_header(f"TRAINING DONE — Best Top-1 = {best_top1:.4f}")
