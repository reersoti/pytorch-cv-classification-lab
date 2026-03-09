import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# -----------------------
# Настройка устройства
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------
# Папка для результатов
# -----------------------
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

# -----------------------
# CIFAR100 transforms
# -----------------------
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409),
                         (0.2673, 0.2564, 0.2762)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409),
                         (0.2673, 0.2564, 0.2762)),
])

# -----------------------
# Датасет локально
# -----------------------
data_root = "./cifar-100-python"
train_dataset = CIFAR100(root=data_root, train=True, download=False, transform=transform_train)
test_dataset = CIFAR100(root=data_root, train=False, download=False, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

classes = train_dataset.classes

# -----------------------
# Модель
# -----------------------
from torchvision.models import resnet18

def build_model(num_classes=100):
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, num_classes)
    return model

model = build_model().to(device)

# -----------------------
# Функция точности
# -----------------------
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k / batch_size).item())
    return res

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -----------------------
# Обучение
# -----------------------
best_val_acc = 0
EPOCHS = 5
checkpoint_path = os.path.join(results_dir, "best_checkpoint.pth")

wrong_images = []
wrong_preds = []
wrong_targets = []

top1_list = []
top5_list = []

for epoch in range(EPOCHS):

    # ---- TRAIN ----
    model.train()
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # ---- VALIDATION ----
    model.eval()
    val_top1 = 0
    val_top5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            batch_size = labels.size(0)

            val_top1 += top1 * batch_size
            val_top5 += top5 * batch_size
            total += batch_size

    val_top1 /= total
    val_top5 /= total

    top1_list.append(val_top1)
    top5_list.append(val_top5)

    print(f"Val Top1: {val_top1:.4f} | Top5: {val_top5:.4f}")

    if val_top1 > best_val_acc:
        best_val_acc = val_top1
        torch.save(model.state_dict(), checkpoint_path)
        print("Saved best checkpoint!")

# -----------------------
# Загрузка лучшей модели
# -----------------------
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

test_top1 = 0
test_top5 = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        top1, top5 = accuracy(outputs, labels, topk=(1, 5))
        batch_size = labels.size(0)
        test_top1 += top1 * batch_size
        test_top5 += top5 * batch_size
        total += batch_size

        # сохраняем неправильные предсказания
        _, preds = outputs.topk(1, 1, True, True)
        preds = preds.squeeze()
        wrong_mask = preds != labels
        for i in range(len(images)):
            if wrong_mask[i]:
                wrong_images.append(images[i].cpu())
                wrong_preds.append(preds[i].item())
                wrong_targets.append(labels[i].item())

test_top1 /= total
test_top5 /= total
print(f"TEST Top-1 Accuracy: {test_top1:.4f}")
print(f"TEST Top-5 Accuracy: {test_top5:.4f}")

# -----------------------
# Визуализация неправильных предсказаний
# -----------------------
def unnormalize(img):
    mean = np.array([0.5071, 0.4865, 0.4409])
    std = np.array([0.2673, 0.2564, 0.2762])
    img = img.permute(1, 2, 0).numpy()
    img = img * std + mean
    return np.clip(img, 0, 1)

N = min(12, len(wrong_images))
plt.figure(figsize=(12, 8))
for i in range(N):
    plt.subplot(3, 4, i+1)
    img = unnormalize(wrong_images[i])
    plt.imshow(img)
    plt.title(f"Pred: {classes[wrong_preds[i]]}\nTrue: {classes[wrong_targets[i]]}", fontsize=8)
    plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "wrong_predictions.png"))
plt.show()

# -----------------------
# График точности по эпохам
# -----------------------
plt.figure()
plt.plot(range(1, EPOCHS+1), top1_list, label="Top-1")
plt.plot(range(1, EPOCHS+1), top5_list, label="Top-5")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy per Epoch")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, "accuracy_per_epoch.png"))
plt.show()
