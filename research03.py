import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import subprocess

# -----------------------
# Настройки
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
data_root = "./data"
batch_size = 128
num_epochs = 40

checkpoint_base = "./checkpoints"
arch_name = "mobilenet_50k_v1"
checkpoint_dir = os.path.join(checkpoint_base, arch_name)
os.makedirs(checkpoint_dir, exist_ok=True)

# -----------------------
# Счётчик параметров
# -----------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------
# Tiny MobileNet (~45.5k params)
# -----------------------
class DWConvBNReLU(nn.Module):
    def __init__(self, channels, k=3, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=k, padding=padding,
                            groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.dw(x)))

class PWConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pw(x)))

class TinyMobileNet50k(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        c1, c2, c3 = 48, 96, 192

        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

        self.block1 = nn.Sequential(
            DWConvBNReLU(c1),
            PWConvBNReLU(c1, c2),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            DWConvBNReLU(c2),
            PWConvBNReLU(c2, c3),
            nn.MaxPool2d(2)
        )

        self.extra = nn.Sequential(
            DWConvBNReLU(c3),
            DWConvBNReLU(c3)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.extra(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# -----------------------
# Трансформации (аугментации включены)
# -----------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5071,0.4867,0.4408],
                         [0.2675,0.2565,0.2761]),
    transforms.RandomErasing(p=0.5, scale=(0.02,0.2))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5071,0.4867,0.4408],
                         [0.2675,0.2565,0.2761])
])

# -----------------------
# Датасеты и лоадеры
# -----------------------
train_set = torchvision.datasets.CIFAR100(
    root=data_root, train=True, download=True, transform=transform_train)

test_set = torchvision.datasets.CIFAR100(
    root=data_root, train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_set, batch_size=batch_size,
                          shuffle=True, num_workers=0, pin_memory=True)

test_loader = DataLoader(test_set, batch_size=batch_size,
                         shuffle=False, num_workers=0, pin_memory=True)

# -----------------------
# Модель / оптимизатор / scheduler
# -----------------------
model = TinyMobileNet50k().to(device)
print("Model params:", count_parameters(model))

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# -----------------------
# Accuracy helper
# -----------------------
@torch.no_grad()
def eval_accuracy(loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    loop = tqdm(loader, desc="Validation", dynamic_ncols=True,
                leave=False, mininterval=0.2)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loop.set_postfix(loss=loss.item())

    return correct / total, total_loss / len(loader)

@torch.no_grad()
def eval_top1_top5(outputs, labels):
    _, pred = outputs.topk(5, dim=1)
    correct1 = (pred[:, 0] == labels).sum().item()
    correct5 = sum(labels[i].item() in pred[i] for i in range(labels.size(0)))
    return correct1, correct5


# -----------------------
# Тренировка
# -----------------------
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                dynamic_ncols=True,
                leave=False,
                mininterval=0.2)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    # validation
    model.eval()
    total = 0
    correct1_total = 0
    correct5_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            total += labels.size(0)

            c1, c5 = eval_top1_top5(outputs, labels)
            correct1_total += c1
            correct5_total += c5

    val_loss_avg = val_loss / total
    top1 = correct1_total / total
    top5 = correct5_total / total

    print(f"[Epoch {epoch+1}] top1={top1:.4f}  top5={top5:.4f}  val_loss={val_loss_avg:.4f}")
# KDE notification
try:
    os.system('paplay /usr/share/sounds/freedesktop/stereo/complete.oga')
except:
    pass

os.system('notify-send "Training Complete" "Model finished all epochs!"')
