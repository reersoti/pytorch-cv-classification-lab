import os, torch, torch.nn as nn, torch.optim as optim
import torchvision.transforms as transforms, torchvision.models as models
from torch.utils.data import DataLoader, random_split, TensorDataset
from experiment_template import *
from tqdm import tqdm

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root, num_classes = "./tnn2025", 46
    train_csv, train_img_dir, test_img_dir = os.path.join(data_root, "train.csv"), os.path.join(data_root, "train/train_256"), os.path.join(data_root, "test/test_256")
    output_dir = "./features_effnet"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load features
    train_path, test_path = os.path.join(output_dir, "train_b0.pt"), os.path.join(output_dir, "test_b0.pt")
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("\n--- Loading EfficientNet-B0 Features ---")
        train_data, test_data = torch.load(train_path, weights_only=True), torch.load(test_path, weights_only=True)
    else:
        print("\n--- Extracting EfficientNet-B0 Features ---")
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).to(device)
        effnet.eval()
        class FeatExtractor(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.features, self.avgpool = m.features, m.avgpool
            def forward(self, x):
                return torch.flatten(self.avgpool(self.features(x)), 1)
        extractor = FeatExtractor(effnet).to(device)
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        def extract(csv, img_dir, name, is_test):
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
        train_data, test_data = extract(train_csv, train_img_dir, "train", False), extract(None, test_img_dir, "test", True)
    
    # Train
    print("\n--- Training EfficientNet-B0 + ResNet1D ---")
    full_ds = TensorDataset(train_data["features"], train_data["labels"])
    ts = int(0.9 * len(full_ds))
    t_db, v_db = random_split(full_ds, [ts, len(full_ds)-ts])
    t_ld, v_ld = DataLoader(t_db, batch_size=128, shuffle=True), DataLoader(v_db, batch_size=128, shuffle=False)
    
    model = ResNet1D(input_dim=1280, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion, num_epochs, best_acc = nn.CrossEntropyLoss(), 50, 0
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
            best_acc = acc; torch.save(model.state_dict(), "efficientnet_b0_resnet1d.pth")
        print(f"Epoch {ep}/{num_epochs} | Val Acc: {acc:.4f} (Best: {best_acc:.4f})")
    
    # Inference
    print("\n--- Generating Submission ---")
    model.load_state_dict(torch.load("efficientnet_b0_resnet1d.pth", weights_only=True))
    model.eval()
    with torch.no_grad():
        all_preds = []
        for i in range(0, len(test_data["features"]), 128):
            _, p = model(test_data["features"][i:i+128].to(device)).max(1)
            all_preds.extend(p.cpu().tolist())
    pd.DataFrame({"id": test_data["labels"].tolist(), "label": all_preds}).sort_values("id").to_csv("submission_effnet_b0_resnet1d.csv", index=False)
    print("Done!")

if __name__ == "__main__":
    main()
