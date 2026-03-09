import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# --- Глобальные настройки ---
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
num_epochs = 40
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
best_top1 = 0.0

# --- Аугментации и датасеты ---
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761))
])

data_root = "./data"
train_set = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

# --- Архитектура ~1M параметров ---
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)

class CIFARResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool1 = nn.MaxPool2d(2)

        self.res1 = ResBlock(256)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.pool2 = nn.MaxPool2d(2)

        self.res2 = ResBlock(512)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 100)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.res1(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.res2(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

model = CIFARResNet().to(device)

# --- Оптимизатор и LR scheduler ---
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# --- Функция обучения и валидации ---
def evaluate(model, loader):
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, pred = outputs.topk(5, 1, True, True)
            total += labels.size(0)
            correct1 += (pred[:, 0] == labels).sum().item()
            correct5 += sum([labels[i] in pred[i] for i in range(labels.size(0))])
    return correct1 / total, correct5 / total

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}", ncols=80):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    scheduler.step()

    top1, top5 = evaluate(model, test_loader)
    print(f"Epoch {epoch} - val loss: {running_loss/len(train_loader.dataset):.4f}, top1: {top1:.4f}, top5: {top5:.4f}")

    # Сохранение лучшего чекпоинта
    if top1 > best_top1:
        best_top1 = top1
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best_1M_epoch{epoch}.pth"))
        print(f"Saved best model: {os.path.join(checkpoint_dir, f'best_1M_epoch{epoch}.pth')}")
