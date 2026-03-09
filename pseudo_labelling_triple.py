import os, torch, torch.nn as nn, torch.optim as optim
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from experiment_template import DenseHead
from tqdm import tqdm

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 46
    
    # 1. Load Triple Features
    print("\n--- Loading Triple Features ---")
    tr_b2 = torch.load("./features_effb2/train_effb2.pt", weights_only=True)
    te_b2 = torch.load("./features_effb2/test_effb2.pt", weights_only=True)
    tr_cn = torch.load("./features_cnext_base/train_cnext_base.pt", weights_only=True)
    te_cn = torch.load("./features_cnext_base/test_cnext_base.pt", weights_only=True)
    tr_d121 = torch.load("./features_dense121/train_dense121.pt", weights_only=True)
    te_d121 = torch.load("./features_dense121/test_dense121.pt", weights_only=True)
    
    train_fused = torch.cat([tr_b2["features"], tr_cn["features"], tr_d121["features"]], dim=1)
    test_fused = torch.cat([te_b2["features"], te_cn["features"], te_d121["features"]], dim=1)
    
    train_labels = tr_b2["labels"]
    test_ids = te_b2["labels"]

    # 2. Get Pseudo Labels using Triple Fusion model
    print("\n--- Generating Pseudo Labels (Threshold=0.95) ---")
    orig_model = DenseHead(input_dim=3456, block_dim=1024, num_classes=num_classes).to(device)
    orig_model.load_state_dict(torch.load("triple_fusion_model.pth", weights_only=True))
    orig_model.eval()
    
    pseudo_features = []
    pseudo_labels = []
    
    with torch.no_grad():
        for i in range(0, len(test_fused), 128):
            batch = test_fused[i:i+128].to(device)
            logits = orig_model(batch)
            probs = F.softmax(logits, dim=1)
            conf, preds = probs.max(1)
            
            # Filter high confidence
            mask = conf > 0.95
            if mask.any():
                pseudo_features.append(test_fused[i:i+128][mask.cpu()])
                pseudo_labels.append(preds[mask])
    
    pseudo_features = torch.cat(pseudo_features)
    pseudo_labels = torch.cat(pseudo_labels)
    print(f"Added {len(pseudo_labels)} pseudo-labeled samples from test set.")

    # 3. Create Combined Dataset
    combined_features = torch.cat([train_fused, pseudo_features.cpu()], dim=0)
    combined_labels = torch.cat([train_labels, pseudo_labels.cpu()], dim=0)
    
    # 4. Train New Best Head
    print("\n--- Training Best Head on Combined Data ---")
    full_ds = TensorDataset(combined_features, combined_labels)
    ts = int(0.9 * len(full_ds))
    t_db, v_db = random_split(full_ds, [ts, len(full_ds)-ts])
    t_ld, v_ld = DataLoader(t_db, batch_size=128, shuffle=True), DataLoader(v_db, batch_size=128, shuffle=False)
    
    model = DenseHead(input_dim=3456, block_dim=1024, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion, num_epochs, best_acc = nn.CrossEntropyLoss(label_smoothing=0.1), 60, 0
    
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
            best_acc = acc; torch.save(model.state_dict(), "triple_pseudo_model.pth")
        print(f"Epoch {ep}/{num_epochs} | Val Acc: {acc:.4f} (Best: {best_acc:.4f})")
    
    # 5. Final Inference
    print("\n--- Generating Final Pseudo-Labeled Submission ---")
    model.load_state_dict(torch.load("triple_pseudo_model.pth", weights_only=True))
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_fused), 128):
            _, p = model(test_fused[i:i+128].to(device)).max(1)
            all_preds.extend(p.cpu().tolist())
            
    pd.DataFrame({"id": test_ids.tolist(), "label": all_preds}).sort_values("id").to_csv("submission_triple_pseudo.csv", index=False)
    print(f"Done! Saved to submission_triple_pseudo.csv")

if __name__ == "__main__":
    main()
