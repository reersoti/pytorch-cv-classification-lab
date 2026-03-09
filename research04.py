import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# -----------------------
# Настройки
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
data_root = "./data"
batch_size = 128
num_epochs = 40
checkpoint_dir = "./checkpoints/micronet_100k"
os.makedirs(checkpoint_dir, exist_ok=True)

# -----------------------
# Helpers
# -----------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def eval_accuracy(model, loader, criterion):
    model.eval()
    correct1, correct5, total, val_loss = 0, 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * imgs.size(0)
        _, pred = outputs.topk(5, 1, True, True)
        total += labels.size(0)
        correct1 += (pred[:,0] == labels).sum().item()
        correct5 += sum([labels[i] in pred[i] for i in range(labels.size(0))])
    return correct1/total, correct5/total, val_loss/total

# -----------------------
# MicroNet-SE Architecture ~100k
# -----------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels//reduction)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(channels//reduction, channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.pool(x).view(b,c)
        s = self.act1(self.fc1(s))
        s = self.sigmoid(self.fc2(s)).view(b,c,1,1)
        return x * s

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, se=True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
        self.se = SEBlock(out_ch) if se else nn.Identity()
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        out = self.dw(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.pw(out)
        out = self.bn2(out)
        out = self.se(out)
        out += self.shortcut(x)
        return self.act(out)

class MicroNetSE100k(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        self.block1 = ResidualBlock(32, 64)
        self.block2 = ResidualBlock(64, 128)
        self.block3 = ResidualBlock(128, 128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

# -----------------------
# Data
# -----------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5071,0.4867,0.4408],[0.2675,0.2565,0.2761]),
    transforms.RandomErasing(p=0.5, scale=(0.02,0.2))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5071,0.4867,0.4408],[0.2675,0.2565,0.2761])
])

train_set = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
test_set  = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_test)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

# -----------------------
# Model / Optimizer / Scheduler / Loss
# -----------------------
model = MicroNetSE100k().to(device)
print("Model params:", count_parameters(model))
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# -----------------------
# Training loop
# -----------------------
best_top1 = 0.0

for epoch in range(1, num_epochs+1):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*imgs.size(0)

    # Validation
    top1, top5, val_loss = eval_accuracy(model, test_loader, criterion)
    print(f"[Epoch {epoch}] top1={top1:.4f}  top5={top5:.4f}  val_loss={val_loss:.4f}")

    # Save best
    if top1 > best_top1:
        best_top1 = top1
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best_top1_{best_top1:.4f}_ep{epoch}.pth"))

# -----------------------
# Notify at the end (cross-platform fallback)
# -----------------------
try:
    import IPython.display as ipd
    ipd.display(ipd.Audio('https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg', autoplay=True))
except:
    print("\a")  # ASCII bell

print(f"\n=== Training complete ===\nBest Top-1: {best_top1:.4f}")
