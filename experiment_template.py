"""
Template for frozen backbone experiments.
This file contains reusable components for all experiments.
"""
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

# Dataset
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

# Heads
class ResBlock1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(x + self.block(x))

class ResNet1D(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512, num_classes=46):
        super().__init__()
        self.init_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(ResBlock1D(hidden_dim), ResBlock1D(hidden_dim), ResBlock1D(hidden_dim))
        self.final_classifier = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = self.init_layer(x)
        x = self.res_blocks(x)
        return self.final_classifier(x)

class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.net(x)

class DenseHead(nn.Module):
    def __init__(self, input_dim=1280, block_dim=512, num_classes=46):
        super().__init__()
        self.layer1 = DenseLayer(input_dim, block_dim)
        self.layer2 = DenseLayer(input_dim + block_dim, block_dim)
        self.layer3 = DenseLayer(input_dim + block_dim * 2, block_dim)
        self.classifier = nn.Linear(input_dim + block_dim * 3, num_classes)
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(torch.cat([x, x1], dim=1))
        x3 = self.layer3(torch.cat([x, x1, x2], dim=1))
        return self.classifier(torch.cat([x, x1, x2, x3], dim=1))

class Conv1DHead(nn.Module):
    def __init__(self, input_dim=1280, num_classes=46):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

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
        x = self.input_proj(x).unsqueeze(1)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)
