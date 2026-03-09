import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# -----------------------
# Настройки
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
data_root = "./data"
batch_size = 128
num_epochs = 40
checkpoint_base = "./checkpoints"
arch_name = "vit_mini_100k"
checkpoint_dir = os.path.join(checkpoint_base, arch_name)
os.makedirs(checkpoint_dir, exist_ok=True)

# -----------------------
# Счётчик параметров
# -----------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------
# Tiny ViT (~100k params)
# -----------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1,2)  # [B, num_patches, embed_dim]
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]  # each [B, heads, N, C//heads]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(dim, hidden_dim, drop=drop)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViTmini100k(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, num_classes=100, embed_dim=48, depth=4, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)
        num_patches = (img_size // patch_size)**2
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,1+num_patches, embed_dim))
        self.pos_drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B,-1,-1)
        x = torch.cat((cls_tokens,x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:,0]
        x = self.head(cls_out)
        return x

# -----------------------
# Data
# -----------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5071,0.4867,0.4408],[0.2675,0.2565,0.2761]),
    transforms.RandomErasing(p=0.5,scale=(0.02,0.2))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5071,0.4867,0.4408],[0.2675,0.2565,0.2761])
])
train_set = torchvision.datasets.CIFAR100(data_root, train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR100(data_root, train=False, download=True, transform=transform_test)
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0)
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=0)

# -----------------------
# Model / Optimizer / Loss / Scheduler
# -----------------------
model = ViTmini100k().to(device)
print("Model params:", count_parameters(model))
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# -----------------------
# Training
# -----------------------
best_top1 = 0.0
for epoch in range(1,num_epochs+1):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    # Validation
    model.eval()
    correct1, correct5, total = 0,0,0
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, pred = outputs.topk(5,1,True,True)
            total += labels.size(0)
            correct1 += (pred[:,0]==labels).sum().item()
            correct5 += sum([labels[i] in pred[i] for i in range(labels.size(0))])
    top1 = correct1/total
    top5 = correct5/total
    print(f"[Epoch {epoch}] top1={top1:.4f}  top5={top5:.4f}  val_loss={val_loss/total:.4f}")

    # Save best
    if top1>best_top1:
        best_top1=top1
        ckpt_name = f"best_{arch_name}_top1_{best_top1:.4f}_ep{epoch}.pth"
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, ckpt_name))

print("\n=== Training complete ===")
print(f"Best Top-1: {best_top1:.4f}")
