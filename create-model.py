import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle

# Настройки
SAMPLE_RATE = 16000 # Частота дискретизации
DURATION = 2.0 # Количество секунд при записи
N_FFT = 512 # Размер окна для преобразования Фурье
COMMANDS = ['время', 'дата', 'алгоритмы'] # Команды
DATA_DIR = 'voice_commands' # Куда сохраняю
BATCH_SIZE = 16  # Размер батча для обучения, то бишь 16 спектрограмм
EPOCHS = 30 # Количество обучений модели
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class VoiceCommandDataset(Dataset):
    def __init__(self, data_dir, commands):
        self.data = []
        self.labels = []
        
        for label_idx, cmd in enumerate(commands):
            cmd_dir = os.path.join(data_dir, cmd)
            for file in os.listdir(cmd_dir):
                if file.endswith('.pkl'): # Загружаю ПКЛьки
                    with open(os.path.join(cmd_dir, file), 'rb') as f:
                        spec = pickle.load(f)
                    # Убедимся, что спектрограмма имеет правильную форму
                    if spec.shape[0] > 0 and spec.shape[1] > 0:
                        self.data.append(spec)
                        self.labels.append(label_idx)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        spec = self.data[idx]
        # Нормализовать и привести к правильному размеру
        spec = (spec - spec.mean()) / (spec.std() + 1e-8) # Вычитаю среднее и делю на квадратное отклонение
        # Все спектрограммы подгоняю к одинаковому размеру
        target_height = N_FFT // 2 + 1
        target_width = int(SAMPLE_RATE * DURATION) // (N_FFT // 4) + 1
        
        if spec.shape != (target_height, target_width):
            from scipy.ndimage import zoom # Масштабирую до нужного размера
            zoom_factors = (target_height / spec.shape[0], target_width / spec.shape[1])
            spec = zoom(spec, zoom_factors)
        
        spec = torch.FloatTensor(spec).unsqueeze(0)  # Добавляю размерность канала, чтобы была совместимость с PyTorch
        return spec, self.labels[idx]

class VoiceCommandCNN(nn.Module):
    def __init__(self, num_classes):
        super(VoiceCommandCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Автоматический расчет размера после сверток и пулинга
        self._to_linear = None
        self._calculate_conv_output_size()
        
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def _calculate_conv_output_size(self):
        # Пробный проход для определения размера
        x = torch.zeros(1, 1, N_FFT // 2 + 1, int(SAMPLE_RATE * DURATION) // (N_FFT // 4) + 1)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        self._to_linear = x.numel() // x.shape[0]
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Выравниваем для полносвязного слоя
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        val_loss, val_acc = evaluate_model(model, val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print('Модель сохранена!')
            
    return model

def evaluate_model(model, loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / len(loader), 100 * correct / total

def main():
    # Проверяем данные
    if not os.path.exists(DATA_DIR) or not any(os.listdir(os.path.join(DATA_DIR, cmd)) for cmd in COMMANDS):
        print("Нет записанных данных. Начните с записи образцов.")
        return
    
    print("Найдены записанные данные. Начинаем обучение...")
    
    # Создаем датасет
    dataset = VoiceCommandDataset(DATA_DIR, COMMANDS)
    print(f"Всего образцов: {len(dataset)}")
    
    # Разделяем на train/val
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Инициализация модели
    model = VoiceCommandCNN(len(COMMANDS)).to(DEVICE)
    print("Архитектура модели:")
    print(model)
    
    # Обучение
    train_model(model, train_loader, val_loader, EPOCHS)
    
    print("Обучение завершено! Лучшая модель сохранена в 'best_model.pth'")

if __name__ == "__main__":
    main()