"""
Universal feature extractor for all new backbones.
Run this once to cache features for all experiments.
"""
import os, torch, torch.nn as nn
import torchvision.transforms as transforms, torchvision.models as models
from torch.utils.data import DataLoader
from experiment_template import KaggleTNN2025Dataset
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
data_root = "./tnn2025"
train_csv = os.path.join(data_root, "train.csv")
train_img_dir = os.path.join(data_root, "train/train_256")
test_img_dir = os.path.join(data_root, "test/test_256")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

backbones = {
    "effb1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.IMAGENET1K_V1, 1280, "./features_effb1"),
    "effb2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.IMAGENET1K_V1, 1408, "./features_effb2"),
    "mobv3": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1, 576, "./features_mobv3"),
    "cnext": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 768, "./features_cnext"),
    "r34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, 512, "./features_r34"),
    "dense121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1, 1024, "./features_dense121"),
    "r101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1, 2048, "./features_r101"),
    "cnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1, 1024, "./features_cnext_base"),
}

def extract_features(name, model_fn, weights, out_dim, output_dir):
    print(f"\n=== Extracting {name} (dim={out_dim}) ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    backbone = model_fn(weights=weights).to(device)
    backbone.eval()
    
    # Create extractor
    if 'eff' in name or 'mobv3' in name:
        extractor = nn.Sequential(backbone.features, backbone.avgpool, nn.Flatten()).to(device)
    elif 'r34' in name or 'r101' in name or 'r50' in name:
        extractor = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten()).to(device)
    elif 'cnext' in name:
        extractor = nn.Sequential(backbone.features, backbone.avgpool, nn.Flatten()).to(device)
    elif 'dense' in name:
        extractor = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d(1), nn.Flatten()).to(device)
    else:
        raise ValueError(f"Unknown backbone type: {name}")
    
    # Extract train
    train_path = os.path.join(output_dir, f"train_{name}.pt")
    if not os.path.exists(train_path):
        print(f"Extracting train...")
        ds = KaggleTNN2025Dataset(train_csv, train_img_dir, transform=transform, is_test=False)
        ld = DataLoader(ds, batch_size=32, shuffle=False)
        af, al = [], []
        with torch.no_grad():
            for imgs, lbls in tqdm(ld, desc="train"):
                af.append(extractor(imgs.to(device)).cpu())
                al.append(lbls)
        torch.save({"features": torch.cat(af), "labels": torch.cat(al)}, train_path)
        print(f"Saved to {train_path}")
    
    # Extract test
    test_path = os.path.join(output_dir, f"test_{name}.pt")
    if not os.path.exists(test_path):
        print(f"Extracting test...")
        ds = KaggleTNN2025Dataset(None, test_img_dir, transform=transform, is_test=True)
        ld = DataLoader(ds, batch_size=32, shuffle=False)
        af, al = [], []
        with torch.no_grad():
            for imgs, lbls in tqdm(ld, desc="test"):
                af.append(extractor(imgs.to(device)).cpu())
                al.append(lbls)
        torch.save({"features": torch.cat(af), "labels": torch.cat(al)}, test_path)
        print(f"Saved to {test_path}")

if __name__ == "__main__":
    for name, (model_fn, weights, out_dim, output_dir) in backbones.items():
        extract_features(name, model_fn, weights, out_dim, output_dir)
    print("\n✅ All features extracted!")
