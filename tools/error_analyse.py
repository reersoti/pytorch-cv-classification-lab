import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
import torch.nn.functional as F
import os

# ===============================
# Подгрузка модели (замени на свой файл)
# ===============================
from resnet_restore import ResNet9_50k  # ваш файл с моделью

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNet9_50k(num_classes=100).to(device)

# Путь к чекпоинту
checkpoint_path = "checkpoints/best_of_all_50k.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
model.eval()

# ===============================
# Параметры модели
# ===============================
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params}")

# ===============================
# CIFAR-100 Dataset
# ===============================
transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
])
test_set = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

# ===============================
# Метаклассы CIFAR-100 (20)
# ===============================
meta_classes = [
    ['apple', 'orange', 'pear', 'sweet_pepper', 'peach'],   # пример структуры, реально взять mapping
    ['maple_tree','oak_tree','palm_tree','pine_tree','willow_tree'],
    ['bed','chair','sofa','table','wardrobe'],
    ['bee','beetle','butterfly','caterpillar','cockroach'],
    ['bottle','bowl','can','cup','plate'],
    ['fox','porcupine','possum','raccoon','skunk'],
    ['crab','lobster','snail','spider','worm'],
    ['baby','boy','girl','man','woman'],
    ['crocodile','dinosaur','lizard','snake','turtle'],
    ['hamster','mouse','rabbit','shrew','squirrel'],
    ['clock','keyboard','lamp','telephone','television'],
    ['bridge','castle','house','road','skyscraper'],
    ['cloud','forest','mountain','plain','sea'],
    ['camel','cattle','chimpanzee','elephant','kangaroo'],
    ['rocket','train','truck','bicycle','motorcycle'],
    ['ball','flag','fountain','guitar','violin'],
    ['car','bus','motorcycle','pickup_truck','train'],
    ['bear','leopard','lion','tiger','wolf'],
    ['fish','shark','dolphin','seal','whale'],
    ['flower','tree','grass','mushroom','cactus']
]

# Преобразуем в mapping: class_idx -> meta_class_idx
# CIFAR-100 идёт по индексу 0-99
class_to_meta = []
for meta_idx, group in enumerate(meta_classes):
    class_to_meta.extend([meta_idx]*5)  # каждая группа содержит 5 классов

assert len(class_to_meta) == 100, "Неправильная длина class_to_meta"

# ===============================
# Сбор статистики ошибок
# ===============================
top1_total, top5_total, count_total = 0, 0, 0
errors_per_class = defaultdict(list)
class_correct = defaultdict(int)
class_total = defaultdict(int)
errors_per_meta = defaultdict(list)
meta_correct = defaultdict(int)
meta_total = defaultdict(int)

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, pred = outputs.topk(5, 1, True, True)

        # Top-1 и Top-5
        top1 = (pred[:,0] == labels).float()
        top1_total += top1.sum().item()
        top5 = torch.any(pred == labels.view(-1,1), dim=1).float()
        top5_total += top5.sum().item()
        count_total += labels.size(0)

        for i in range(labels.size(0)):
            true_cls = labels[i].item()
            pred_cls = pred[i,0].item()

            if true_cls >= len(class_to_meta) or pred_cls >= len(class_to_meta):
                continue  # пропускаем некорректные индексы

            # Ошибки по классам
            class_total[true_cls] += 1
            if true_cls != pred_cls:
                errors_per_class[true_cls].append(pred_cls)
            else:
                class_correct[true_cls] += 1

            # Ошибки по метаклассам
            meta_cls = class_to_meta[true_cls]
            pred_meta = class_to_meta[pred_cls]
            meta_total[meta_cls] += 1
            if meta_cls != pred_meta:
                errors_per_meta[meta_cls].append(pred_meta)
            else:
                meta_correct[meta_cls] += 1

# ===============================
# Итоги
# ===============================
top1_acc = top1_total / count_total
top5_acc = top5_total / count_total
print(f"\nTop-1 Accuracy: {top1_acc:.4f}")
print(f"Top-5 Accuracy: {top5_acc:.4f}\n")

print("Top errors per class:")
for cls in sorted(errors_per_class.keys()):
    print(f"Class {cls}: {len(errors_per_class[cls])} errors, Top-1 correct {class_correct[cls]}/{class_total[cls]}")

print("\nTop errors per meta-class:")
for mc in sorted(errors_per_meta.keys()):
    print(f"Meta-class {mc}: {len(errors_per_meta[mc])} errors, Top-1 correct {meta_correct[mc]}/{meta_total[mc]}")

# ===============================
# Топ ошибок: кого с кем путает
# ===============================
print("\nTop confusion per class (most common wrong prediction):")
for cls in sorted(errors_per_class.keys()):
    if errors_per_class[cls]:
        counter = Counter(errors_per_class[cls])
        most_common_pred, count_err = counter.most_common(1)[0]
        print(f"Class {cls} is mostly confused with Class {most_common_pred} ({count_err} times)")
