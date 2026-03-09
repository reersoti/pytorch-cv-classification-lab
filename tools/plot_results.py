import matplotlib.pyplot as plt

# =========================
# Данные экспериментов
# =========================
models = [
    # Малые CNN
    {"name": "Simple CNN ~50k (pre-opt)", "params": 49_159, "top1": 65.77, "top5": 89.37},
    {"name": "Simple CNN ~50k (opt)",     "params": 49_159, "top1": 59.85, "top5": 86.00},

    # Mobile / Micro
    {"name": "MobileNet ~50k",            "params": 50_000, "top1": 50.67, "top5": 79.98},
    {"name": "MicroNet-SE ~72k",           "params": 72_820, "top1": 47.00, "top5": 77.36},
    {"name": "ViT-mini ~86k",              "params": 86_356, "top1": 46.49, "top5": 76.90},
    {"name": "MicroNet ~100k",             "params": 100_000,"top1": 51.00, "top5": 80.00},

    # Transformer
    {"name": "ViT-mini ~557k",             "params": 557_796,"top1": 50.83, "top5": 79.91},

    # Большая модель
    {"name": "ResNet-like ~1M",            "params": 1_000_000, "top1": 72.39, "top5": 92.44},
]

# =========================
# Подготовка данных
# =========================
names = [m["name"] for m in models]
top1 = [m["top1"] for m in models]
top5 = [m["top5"] for m in models]

# =========================
# График Top-1
# =========================
plt.figure(figsize=(14, 6))
bars = plt.bar(names, top1)

plt.ylim(0, 100)
plt.ylabel("Top-1 Accuracy (%)")
plt.title("CIFAR-100 — Top-1 Accuracy для разных архитектур")
plt.xticks(rotation=30, ha="right")

# подписи значений
for bar, val in zip(bars, top1):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 1,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.savefig("cifar100_top1_bar.png", dpi=300)
plt.close()

# =========================
# График Top-5
# =========================
plt.figure(figsize=(14, 6))
bars = plt.bar(names, top5)

plt.ylim(0, 100)
plt.ylabel("Top-5 Accuracy (%)")
plt.title("CIFAR-100 — Top-5 Accuracy для разных архитектур")
plt.xticks(rotation=30, ha="right")

# подписи значений
for bar, val in zip(bars, top5):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 1,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.savefig("cifar100_top5_bar.png", dpi=300)
plt.close()

print("Готово ✅")
print("Файлы сохранены:")
print(" - cifar100_top1_bar.png")
print(" - cifar100_top5_bar.png")
