# pytorch_cnn_from_scratch.py
import zipfile
from PIL import Image
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
import random

# ------- Настройки -------
ZIP_PATH = 'cars_bikes_data.zip'   # путь к вашему zip
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3
TRAIN_SIZE = 0.8
RANDOM_STATE = 42
PATIENCE = 5  # ранняя остановка по валидационной потере
MODEL_CHECKPOINT = 'best_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

# зафиксируем сиды для воспроизводимости
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# ------- Загрузка изображений из zip -------
def load_images_from_zip(zip_path):
    images = []
    labels = []
    with zipfile.ZipFile(zip_path, 'r') as archive:
        for file in archive.namelist():
            # игнорируем директории
            if file.endswith('/'):
                continue
            if file.startswith('Car-Bike-Dataset/Bike') and (file.lower().endswith('.jpg') or file.lower().endswith('.jpeg') or file.lower().endswith('.png')):
                label = 0
            elif file.startswith('Car-Bike-Dataset/Car') and (file.lower().endswith('.jpg') or file.lower().endswith('.jpeg') or file.lower().endswith('.png')):
                label = 1
            else:
                continue
            with archive.open(file) as file_handle:
                try:
                    image = Image.open(file_handle).convert('RGB')
                except Exception as e:
                    print("Не удалось открыть изображение:", file, e)
                    continue
                image = image.resize(IMAGE_SIZE)
                image_array = np.array(image).astype(np.float32) / 255.0  # нормировка 0..1
                images.append(image_array)
                labels.append(label)
    X = np.array(images)  # shape (N, H, W, C)
    Y = np.array(labels)
    return X, Y

print("Загрузка данных из ZIP...")
X, Y = load_images_from_zip(ZIP_PATH)
print(f"Загружено изображений: {len(X)}")

# ------- Стратифицированное разбиение -------
sss = StratifiedShuffleSplit(n_splits=1, train_size=TRAIN_SIZE, random_state=RANDOM_STATE)
train_idx, test_idx = next(sss.split(X, Y))
x_train, x_test = X[train_idx], X[test_idx]
y_train, y_test = Y[train_idx].astype(np.float32), Y[test_idx].astype(np.float32)

print(f"Train: {len(x_train)}, Test: {len(x_test)}")

# ------- Dataset и DataLoader (на основе numpy массивов) -------
class NumpyImageDataset(Dataset):
    def __init__(self, images_np, labels_np, augment=False):
        """
        images_np: numpy array (N, H, W, C), float32 0..1
        labels_np: numpy array (N,)
        """
        self.images = images_np
        self.labels = labels_np
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # H,W,C
        # трансформируем в C,H,W
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        # простая аугментация при обучении
        if self.augment:
            # случайный горизонтальный флип
            if random.random() > 0.5:
                img = np.flip(img, axis=2).copy()
            # случайный сдвиг ±5 пикселей
            # (простейшая версия, без заполнения)
        label = self.labels[idx].astype(np.float32)
        return torch.tensor(img), torch.tensor(label)

train_dataset = NumpyImageDataset(x_train, y_train, augment=True)
test_dataset = NumpyImageDataset(x_test, y_test, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ------- Определение модели (CNN) вручную в PyTorch -------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # аналогично вашей архитектуре: расширяем фильтры
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # сохраняет размер
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # После 4 пуллингов сначально 128x128 -> 8x8 (128 / 2 /2 /2 /2 = 8)
        self.flatten_size = 128 * (IMAGE_SIZE[0] // 16) * (IMAGE_SIZE[1] // 16)
        # Note: IMAGE_SIZE 128 -> after 4 pools (factor 16) -> 8

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.fc_out = nn.Linear(64, 1)  # будем использовать BCEWithLogitsLoss

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc_out(x)  # logits, shape (batch,1)
        return logits.squeeze(1)  # shape (batch,)

# ------- Функции для обучения и валидации -------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)  # logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# ------- Инициализация модели, оптимизатора, критерия -------
model = SimpleCNN().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()  # принимает logits
optimizer = optim.Adam(model.parameters(), lr=LR)

# ------- Ранняя остановка и чекпойнт -------
best_val_loss = float('inf')
epochs_no_improve = 0

print("Начинаем обучение на устройстве:", DEVICE)
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

    print(f"Epoch {epoch:02d} | Train loss: {train_loss:.4f} acc: {train_acc*100:.2f}% | Val loss: {val_loss:.4f} acc: {val_acc*100:.2f}%")

    # чекпойнт
    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, MODEL_CHECKPOINT)
        print(f"  Сохранён лучший чекпойнт (val_loss={val_loss:.4f}) -> {MODEL_CHECKPOINT}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Ранняя остановка: нет улучшений {PATIENCE} эпох(ы). Прерываем обучение.")
            break

# ------- Загрузка лучшей модели и оценка на тесте -------
if os.path.exists(MODEL_CHECKPOINT):
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Загружен лучший чекпойнт для финальной оценки.")

test_loss, test_acc = validate(model, val_loader, criterion, DEVICE)
print(f"Точность предсказания на тестовых данных : {test_acc * 100:.5f}%")
