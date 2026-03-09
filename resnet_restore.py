import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# ===============================
# Residual Block 3 12 2025 9 51
# ===============================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)

# ===============================
# ResNet9 Reduced ≤50k params
# ===============================
class ResNet9_50k(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.res1 = ResidualBlock(128)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 255, 3, padding=1, bias=False),
            nn.BatchNorm2d(255),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.res2 = ResidualBlock(255)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(255, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.res2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ===============================
# Dataset & Dataloader
# ===============================
def get_loaders(batch_size=128, num_workers=0):
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    train_set = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

# ===============================
# Accuracy top-1 / top-5
# ===============================
@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k / batch_size).item())
    return res

# ===============================
# Training loop + EarlyStopping
# ===============================
def train_model(
        epochs=40,
        lr=1e-3,
        patience=7,
        device="cuda" if torch.cuda.is_available() else "cpu"
):
    train_loader, test_loader = get_loaders()
    
    # --- проверка первых батчей ---
    x, y = next(iter(train_loader))
    print(f"First train batch shape: {x.shape}, labels: {y.shape}")

    model = ResNet9_50k().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    wait = 0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / len(train_loader))

        # Validation
        model.eval()
        top1_sum, count = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            t1, = accuracy(outputs, labels, topk=(1,))
            top1_sum += t1 * images.size(0)
            count += images.size(0)
        val_top1 = top1_sum / count
        print(f"Validation top-1: {val_top1:.4f}")

        # Save best checkpoint
        if val_top1 > best_acc:
            best_acc = val_top1
            torch.save(model.state_dict(), "checkpoints/best_50k.pth")
            print("✓ Saved new BEST model")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # Final evaluation
    model.load_state_dict(torch.load("checkpoints/best_50k.pth", weights_only=True))
    top1_sum, top5_sum, count = 0, 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        t1, t5 = accuracy(outputs, labels, topk=(1,5))
        top1_sum += t1 * images.size(0)
        top5_sum += t5 * images.size(0)
        count += images.size(0)
    final_top1 = top1_sum / count
    final_top5 = top5_sum / count
    print(f"\n=== Final Results ===\nTop-1: {final_top1:.4f}, Top-5: {final_top5:.4f}")

    return model, final_top1, final_top5

# ===============================
# Entry point
# ===============================
if __name__ == "__main__":
    model, top1, top5 = train_model()
