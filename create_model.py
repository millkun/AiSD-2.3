import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    spectrograms = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for pkl_file in os.listdir(class_dir):
            if pkl_file.endswith('.pkl'):
                with open(os.path.join(class_dir, pkl_file), 'rb') as f:
                    spec = pickle.load(f)
                spectrograms.append(spec)
                labels.append(class_idx)
    
    return np.array(spectrograms), np.array(labels), class_names







class ResNetSpectrogram(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=0, bias=False)
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        #print(x.shape)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        #print(x.shape)
        
        x = self.adaptive_pool(x)
        #print(x.shape)
        
        x = torch.flatten(x, 1)
        #print(x.shape)

        x = self.fc(x)
        #print(x.shape)
        return x

def prepare_loaders(X, y, test_size=0.2, batch_size=32):
    #X = (X - X.mean()) / X.std()
    X = torch.FloatTensor(X).unsqueeze(1)
    y = torch.LongTensor(y)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size
    )
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, classes, epochs=15, lr=0.0001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    train_loss_data = [0]
    val_loss_data = [0]
    train_acc_data = [0]
    val_acc_data = [0]
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss, correct = 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()


        model.eval()
        val_loss, val_correct = 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Метрики
        train_loss /= len(train_loader)
        train_acc = correct / len(train_loader.dataset)
        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)
        train_loss_data.append(train_loss)
        train_acc_data.append(train_acc)
        val_loss_data.append(val_loss)
        val_acc_data.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step(val_acc)

    return train_loss_data, train_acc_data, val_loss_data, val_acc_data

if __name__ == "__main__":
    DATA_DIR = "./data/norm/"
    BATCH_SIZE = 20
    EPOCHS = 15
    
    X, y, classes = load_data(DATA_DIR)
    print(f'размер датасета {X.shape[0]}')
    train_loader, val_loader = prepare_loaders(X, y, batch_size=BATCH_SIZE)
    model = ResNetSpectrogram(num_classes=len(classes))
    print(classes)
    print('обучение')
    train_loss, train_acc, val_loss, val_acc = train_model(model, train_loader, val_loader, classes, epochs=EPOCHS)
