import os, torch, torch.nn as nn, torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, random_split, TensorDataset
from experiment_template import DenseHead
from tqdm import tqdm

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 46
    
    # Пути к извлеченным признакам
    dir_b2 = "./features_effb2"
    dir_cnext = "./features_cnext_base"
    
    print("\n--- Loading Features for Fusion ---")
    # Загружаем EffB2
    train_b2 = torch.load(os.path.join(dir_b2, "train_effb2.pt"), weights_only=True)
    test_b2 = torch.load(os.path.join(dir_b2, "test_effb2.pt"), weights_only=True)
    
    # Загружаем ConvNeXt-Base
    train_cnext = torch.load(os.path.join(dir_cnext, "train_cnext_base.pt"), weights_only=True)
    test_cnext = torch.load(os.path.join(dir_cnext, "test_cnext_base.pt"), weights_only=True)
    
    # Склеиваем признаки (Concatenation)
    # Train: [N, 1408] + [N, 1024] -> [N, 2432]
    # Важно: предполагаем, что порядок картинок совпадает (оба скрипта сортировали filenames)
    train_fused = torch.cat([train_b2["features"], train_cnext["features"]], dim=1)
    test_fused = torch.cat([test_b2["features"], test_cnext["features"]], dim=1)
    
    labels = train_b2["labels"]
    test_ids = test_b2["labels"]
    
    print(f"Fused features dimension: {train_fused.shape[1]}")
    
    # Train
    print("\n--- Training Fusion (EffB2 + CNext) + DenseHead ---")
    full_ds = TensorDataset(train_fused, labels)
    ts = int(0.9 * len(full_ds))
    t_db, v_db = random_split(full_ds, [ts, len(full_ds)-ts])
    t_ld, v_ld = DataLoader(t_db, batch_size=128, shuffle=True), DataLoader(v_db, batch_size=128, shuffle=False)
    
    # Входная размерность: 1408 + 1024 = 2432
    model = DenseHead(input_dim=2432, block_dim=1024, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)
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
            best_acc = acc; torch.save(model.state_dict(), "fusion_v1_model.pth")
        print(f"Epoch {ep}/{num_epochs} | Val Acc: {acc:.4f} (Best: {best_acc:.4f})")
    
    # Inference
    print("\n--- Generating Fused Submission ---")
    model.load_state_dict(torch.load("fusion_v1_model.pth", weights_only=True))
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_fused), 128):
            _, p = model(test_fused[i:i+128].to(device)).max(1)
            all_preds.extend(p.cpu().tolist())
            
    pd.DataFrame({"id": test_ids.tolist(), "label": all_preds}).sort_values("id").to_csv("submission_fusion_v1.csv", index=False)
    print("Done! Result saved to submission_fusion_v1.csv")

if __name__ == "__main__":
    main()
