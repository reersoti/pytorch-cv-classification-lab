import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ==========================================
# 1. Dataset Class
# ==========================================
class KaggleTNN2025Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_test=False):
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        if not is_test:
            self.data = pd.read_csv(csv_file)
        else:
            self.filenames = sorted(os.listdir(img_dir))

    def __len__(self):
        if not self.is_test: return len(self.data)
        return len(self.filenames)

    def __getitem__(self, idx):
        if not self.is_test:
            img_name = self.data.iloc[idx, 1]
            label = self.data.iloc[idx, 2]
            img_path = os.path.join(self.img_dir, img_name)
        else:
            img_name = self.filenames[idx]
            img_path = os.path.join(self.img_dir, img_name)
            try: img_id = int(img_name.split('_')[1].split('.')[0])
            except: img_id = idx
            label = img_id
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, label

# ==========================================
# 2. Transformer Head
# ==========================================
class TransformerHead(nn.Module):
    def __init__(self, input_dim=1280, d_model=128, nhead=8, num_layers=2, num_classes=46):
        super().__init__()
        # EfficientNet-B0 output is 1280 (after avgpool)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.num_tokens = 1 # We use the pooled vector as a single token for simplicity in fine-tuning
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x is [B, 1280]
        x = self.input_proj(x).unsqueeze(1) # [B, 1, 128]
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# ==========================================
# 3. Full Model (EfficientNet + Transformer)
# ==========================================
class PizzaBeater(nn.Module):
    def __init__(self, num_classes=46):
        super().__init__()
        # Загружаем EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Отрезаем классификатор
        self.features = self.backbone.features
        self.avgpool = self.backbone.avgpool
        
        # Наша голова
        self.head = TransformerHead(input_dim=1280, num_classes=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = "./tnn2025"
    train_csv = os.path.join(data_root, "train.csv")
    train_img_dir = os.path.join(data_root, "train/train_256")
    test_img_dir = os.path.join(data_root, "test/test_256")
    
    num_classes = 46
    batch_size = 32
    num_epochs = 15

    # 4.1 Data Augmentation (Более мощная для борьбы за топ)
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 4.2 Split
    full_dataset = KaggleTNN2025Dataset(train_csv, train_img_dir, transform=transform_train)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    val_subset.dataset.transform = transform_val # Меняем трансформ для валидации

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4.3 Model initialization
    model = PizzaBeater(num_classes=num_classes).to(device)
    
    # Сначала немного заморозим бэкбон, чтобы голова "прогрелась" (опционально, но здесь сделаем низкий LR для всех)
    # Настраиваем разные LR: для головы побольше, для бэкбона поменьше
    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-4}, 
        {'params': model.head.parameters(), 'lr': 5e-4}
    ], weight_decay=1e-2)

    # Label Smoothing Loss (дает отличный буст!)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 4.4 Training Loop
    print(f"\n--- Start PizzaBeater Training (Fine-tuning EfficientNet) ---")
    best_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, p = outputs.max(1)
                total += labels.size(0)
                correct += p.eq(labels).sum().item()
        
        acc = correct / total
        print(f"Val Acc: {acc:.4f} (Best: {max(acc, best_acc):.4f})")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_pizza_beater.pth")
        scheduler.step()

    # 4.5 Inference
    print("\n--- Generating Submission (PizzaBeater) ---")
    model.load_state_dict(torch.load("best_pizza_beater.pth", weights_only=True))
    model.eval()
    test_ds = KaggleTNN2025Dataset(None, test_img_dir, transform=transform_val, is_test=True)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    results = []
    with torch.no_grad():
        for imgs, ids in tqdm(test_ld, desc="Inference"):
            outs = model(imgs.to(device))
            _, p = outs.max(1)
            for i in range(len(ids)):
                results.append({"id": ids[i].item(), "label": p[i].item()})
    
    pd.DataFrame(results).sort_values("id").to_csv("submission_efficientnet-b0_transformer_no_frost.csv", index=False)
    print("Done! Submission saved to submission_efficientnet-b0_transformer_no_frost.csv")

if __name__ == "__main__":
    main()
