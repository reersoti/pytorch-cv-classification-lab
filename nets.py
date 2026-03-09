import torch
import torch.nn as nn

class SmallNetwork(nn.Module):
    """
    Очень маленькая сеть (~500 параметров).

    Структура:
      Conv(3→4, 3×3) → ReLU → MaxPool(4)
      Conv(4→8, 3×3) → ReLU → AdaptiveAvgPool(1)
      Linear(8 → 10)

    Используется при сильных ограничениях вычислительного бюджета.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),  # 64 → 16
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # результат: 8×1×1
        )
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class MediumNetwork(nn.Module):
    """
    Средняя сеть (~4600 параметров).

    Структура:
      Conv(3→6) → ReLU → MaxPool(2)
      Conv(6→12) → ReLU → MaxPool(2)
      Conv(12→32) → ReLU → AdaptiveAvgPool(1)
      Linear(32 → 10)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 → 32
            nn.Conv2d(6, 12, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 → 16
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # результат: 32×1×1
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class LargeNetwork(nn.Module):
    """
    Более крупная сеть (~9500 параметров).

    Структура:
      Conv(3→8)  → ReLU → MaxPool(2)
      Conv(8→16) → ReLU → MaxPool(2)
      Conv(16→48) → ReLU → AdaptiveAvgPool(1)
      Linear(48→20) → ReLU → Linear(20→10)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 → 32
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 → 16
            nn.Conv2d(16, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # результат: 48×1×1
        )
        self.classifier = nn.Sequential(
            nn.Linear(48, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """Возвращает количество обучаемых параметров модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("SmallNetwork params:", count_parameters(SmallNetwork()))
    print("MediumNetwork params:", count_parameters(MediumNetwork()))
    print("LargeNetwork params:", count_parameters(LargeNetwork()))

    x = torch.randn(2, 3, 64, 64)
    print("Small forward:", SmallNetwork()(x).shape)
    print("Medium forward:", MediumNetwork()(x).shape)
    print("Large forward:", LargeNetwork()(x).shape)
