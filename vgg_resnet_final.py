import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
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
        if not self.is_test:
            return len(self.data)
        return len(self.filenames)

    def __getitem__(self, idx):
        if not self.is_test:
            img_name = self.data.iloc[idx, 1]
            label = self.data.iloc[idx, 2]
            img_path = os.path.join(self.img_dir, img_name)
        else:
            img_name = self.filenames[idx]
            img_path = os.path.join(self.img_dir, img_name)
            try:
                img_id = int(img_name.split('_')[1].split('.')[0])
            except:
                img_id = idx
            label = img_id

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ==========================================
# 2. ResNet Head Architecture (1D)
# ==========================================
class ResBlock1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class ResNet1D(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=512, num_classes=46):
        super().__init__()
        self.init_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(
            ResBlock1D(hidden_dim),
            ResBlock1D(hidden_dim),
            ResBlock1D(hidden_dim)
        )
        self.final_classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.init_layer(x)
        x = self.res_blocks(x)
        return self.final_classifier(x)

# ==========================================
# 3. Execution Pipeline
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = "./tnn2025"
    train_csv = os.path.join(data_root, "train.csv")
    train_img_dir = os.path.join(data_root, "train/train_256")
    test_img_dir = os.path.join(data_root, "test/test_256")
    output_dir = "./features"
    os.makedirs(output_dir, exist_ok=True)
    
    num_classes = 46
    batch_size = 32
    
    # --- PHASE 1: FEATURE EXTRACTION ---
    print("\n--- PHASE 1: Extracting VGG16 fc7 Features ---")
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    vgg.eval()
    
    # Extractor (up to fc7 + ReLU)
    extractor = nn.Sequential(
        vgg.features,
        vgg.avgpool,
        nn.Flatten(),
        *list(vgg.classifier.children())[:5]
    ).to(device)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def do_extract(csv, img_dir, name, is_test=False):
        path = os.path.join(output_dir, f"{name}_fc7.pt")
        if os.path.exists(path):
            print(f"Skipping extraction for {name}, file exists.")
            return torch.load(path, weights_only=True)
        
        ds = KaggleTNN2025Dataset(csv, img_dir, transform=transform, is_test=is_test)
        ld = DataLoader(ds, batch_size=batch_size, shuffle=False)
        all_f, all_l = [], []
        with torch.no_grad():
            for imgs, lbls in tqdm(ld, desc=name):
                f = extractor(imgs.to(device)).cpu()
                all_f.append(f)
                all_l.append(lbls)
        res = {"features": torch.cat(all_f), "labels": torch.cat(all_l)}
        torch.save(res, path)
        return res

    train_data = do_extract(train_csv, train_img_dir, "train")
    test_data = do_extract(None, test_img_dir, "test", is_test=True)
    
    # --- PHASE 2: TRAINING ---
    print("\n--- PHASE 2: Training ResNet Head ---")
    full_ds = TensorDataset(train_data["features"], train_data["labels"])
    train_size = int(0.9 * len(full_ds))
    t_db, v_db = random_split(full_ds, [train_size, len(full_ds)-train_size])
    
    t_ld = DataLoader(t_db, batch_size=128, shuffle=True)
    v_ld = DataLoader(v_db, batch_size=128, shuffle=False)
    
    model = ResNet1D(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 30 # Сократим для примера, вы уже видели результат
    
    best_acc = 0
    for ep in range(1, num_epochs + 1):
        model.train()
        for x, y in t_ld:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(); model(x); loss = criterion(model(x), y); loss.backward(); optimizer.step()
        
        model.eval(); corr = 0
        with torch.no_grad():
            for x, y in v_ld:
                _, p = model(x.to(device)).max(1); corr += p.eq(y.to(device)).sum().item()
        acc = corr / len(v_db)
        if acc > best_acc:
            best_acc = acc; torch.save(model.state_dict(), "best_model_unified.pth")
        print(f"Epoch {ep}/{num_epochs} | Val Acc: {acc:.4f}")

    # --- PHASE 3: INFERENCE ---
    print("\n--- PHASE 3: Generating Submission ---")
    model.load_state_dict(torch.load("best_model_unified.pth", weights_only=True))
    model.eval()
    
    with torch.no_grad():
        out = model(test_data["features"].to(device))
        _, preds = out.max(1)
    
    sub = pd.DataFrame({"id": test_data["labels"].tolist(), "label": preds.cpu().tolist()}).sort_values("id")
    sub.to_csv("submission_unified.csv", index=False)
    print("Done! Result saved in submission_unified.csv")

if __name__ == "__main__":
    main()
