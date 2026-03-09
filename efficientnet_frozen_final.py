import os
import torch
import torch.nn as nn
import torch.optim as optim
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
# 2. TransformerHead Architecture
# ==========================================
class TransformerHead(nn.Module):
    def __init__(self, input_dim=1280, d_model=128, nhead=8, num_layers=4, num_classes=46):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.num_tokens = 1
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
# 3. Execution Pipeline
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = "./tnn2025"
    train_csv = os.path.join(data_root, "train.csv")
    train_img_dir = os.path.join(data_root, "train/train_256")
    test_img_dir = os.path.join(data_root, "test/test_256")
    output_dir = "./features_effnet" 
    os.makedirs(output_dir, exist_ok=True)
    num_classes = 46

    # --- PHASE 1: Data Check ---
    train_path = os.path.join(output_dir, "train_b0.pt")
    test_path = os.path.join(output_dir, "test_b0.pt")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("\n--- PHASE 1: Loading existing EfficientNet-B0 Features ---")
        train_data = torch.load(train_path, weights_only=True)
        test_data = torch.load(test_path, weights_only=True)
    else:
        print("\n--- PHASE 1: Extracting EfficientNet-B0 Features ---")
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).to(device)
        effnet.eval()
        # Features + global pooling
        class FeatureExtractor(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.features = original_model.features
                self.avgpool = original_model.avgpool
            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                return torch.flatten(x, 1)

        extractor = FeatureExtractor(effnet).to(device)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        def run_ext(csv, img_dir, name, is_test):
            ds = KaggleTNN2025Dataset(csv, img_dir, transform=transform, is_test=is_test)
            ld = DataLoader(ds, batch_size=32, shuffle=False)
            af, al = [], []
            with torch.no_grad():
                for imgs, lbls in tqdm(ld, desc=name):
                    af.append(extractor(imgs.to(device)).cpu())
                    al.append(lbls)
            res = {"features": torch.cat(af), "labels": torch.cat(al)}
            torch.save(res, os.path.join(output_dir, f"{name}_b0.pt"))
            return res
        
        train_data = run_ext(train_csv, train_img_dir, "train", False)
        test_data = run_ext(None, test_img_dir, "test", True)

    # --- PHASE 2: TRAINING ---
    print("\n--- PHASE 2: Training TransformerHead (Frozen Backbone: EfficientNet-B0) ---")
    full_ds = TensorDataset(train_data["features"], train_data["labels"])
    ts = int(0.9 * len(full_ds))
    t_db, v_db = random_split(full_ds, [ts, len(full_ds)-ts])
    t_ld = DataLoader(t_db, batch_size=128, shuffle=True)
    v_ld = DataLoader(v_db, batch_size=128, shuffle=False)
    
    model = TransformerHead(input_dim=1280, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 50 
    
    best_acc = 0
    for ep in range(1, num_epochs + 1):
        model.train()
        for x, y in t_ld:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step()
        model.eval(); corr = 0
        with torch.no_grad():
            for x, y in v_ld:
                _, p = model(x.to(device)).max(1); corr += p.eq(y.to(device)).sum().item()
        acc = corr / len(v_db)
        if acc > best_acc:
            best_acc = acc; torch.save(model.state_dict(), "best_model_effnet_frozen.pth")
        print(f"Epoch {ep}/{num_epochs} | Val Acc: {acc:.4f} (Best: {best_acc:.4f})")

    # --- PHASE 3: INFERENCE ---
    print("\n--- PHASE 3: Generating Submission ---")
    model.load_state_dict(torch.load("best_model_effnet_frozen.pth", weights_only=True))
    model.eval()
    with torch.no_grad():
        test_feats = test_data["features"]
        all_preds = []
        for i in range(0, len(test_feats), 128):
            batch = test_feats[i : i+128].to(device)
            _, p = model(batch).max(1)
            all_preds.extend(p.cpu().tolist())
    sub = pd.DataFrame({"id": test_data["labels"].tolist(), "label": all_preds}).sort_values("id")
    sub.to_csv("submission_effnet_frozen.csv", index=False)
    print("Done! Result saved in submission_effnet_frozen.csv")

if __name__ == "__main__":
    main()
