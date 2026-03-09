import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# ===============================
# Residual Block
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
# ResNet9 ≤50k
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
# CIFAR-100 loaders (TEST only)
# ===============================
def get_test_loader(batch_size=128):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.507, 0.487, 0.441),
                    (0.267, 0.256, 0.276)),
    ])
    test_set = torchvision.datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    return DataLoader(test_set, batch_size=batch_size, shuffle=False)

# ===============================
# Accuracy top-1 / top-5 (FIXED)
# ===============================
@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1)   # [B, k]
    pred = pred.t()                      # [k, B]
    correct = pred.eq(target.view(1, -1))

    res = []
    for k in topk:
        correct_k = correct[:k].any(dim=0).float().mean().item()
        res.append(correct_k)
    return res

# ===============================
# MAIN
# ===============================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model = ResNet9_50k().to(device)

    print("Loading checkpoint...")
    model.load_state_dict(
        torch.load("checkpoints/best_50k.pth", map_location=device)
    )
    model.eval()

    test_loader = get_test_loader()

    top1_sum, top5_sum, count = 0, 0, 0

    print("Running evaluation...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            t1, t5 = accuracy(outputs, labels, topk=(1, 5))
            top1_sum += t1 * images.size(0)
            top5_sum += t5 * images.size(0)
            count += images.size(0)

    print("\n=== RESULTS FROM CHECKPOINT ===")
    print(f"Top-1 Accuracy: {top1_sum / count:.4f}")
    print(f"Top-5 Accuracy: {top5_sum / count:.4f}")

if __name__ == "__main__":
    main()
