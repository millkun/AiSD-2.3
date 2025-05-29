import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from scipy.signal import stft
import librosa
import threading
import time

# Настройки
SAMPLE_RATE = 16000
DURATION = 2.0
N_FFT = 512
COMMANDS = ['время', 'дата', 'алгоритмы']
MODEL_PATH = 'best_model.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Класс модели (должен соответствовать обученной модели)
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
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Загрузка модели
def load_model():
    model = VoiceCommandCNN(len(COMMANDS)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# Преобразование аудио в спектрограмму
def audio_to_spectrogram(audio_data):
    # Нормализация аудио
    audio_data = np.array(audio_data, dtype=np.float32)
    audio_data = (audio_data - audio_data.mean()) / (audio_data.std() + 1e-8)
    
    # Дополнение или обрезка до нужной длины
    target_length = int(SAMPLE_RATE * DURATION)
    if len(audio_data) > target_length:
        audio_data = audio_data[:target_length]
    elif len(audio_data) < target_length:
        audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), 'constant')
    
    # Вычисление спектрограммы
    _, _, spec = stft(audio_data, fs=SAMPLE_RATE, nperseg=N_FFT)
    spec = np.abs(spec)
    
    # Масштабирование до нужного размера
    target_height = N_FFT // 2 + 1
    target_width = int(SAMPLE_RATE * DURATION) // (N_FFT // 4) + 1
    
    if spec.shape != (target_height, target_width):
        from scipy.ndimage import zoom
        zoom_factors = (target_height / spec.shape[0], target_width / spec.shape[1])
        spec = zoom(spec, zoom_factors)
    
    return spec

# Инициализация Flask приложения
app = Flask(__name__)
model = load_model()

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        # Получение JSON данных
        data = request.get_json()
        
        if not data or 'audio' not in data:
            return jsonify({'error': 'No audio data received'}), 400
        
        audio_data = data['audio']
        
        # Проверка уровня звука (если нужно)
        avg_level = np.mean(np.abs(audio_data))
        if avg_level < 100:  # Пороговое значение
            return jsonify({'status': 'silence', 'debug': {'avg_level': avg_level}})
        
        # Преобразование аудио в спектрограмму
        spectrogram = audio_to_spectrogram(audio_data)
        
        # Подготовка данных для модели
        input_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Распознавание команды
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs.data, 1)
            command = COMMANDS[predicted.item()]
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # Формирование ответа
        response = {
            'command': command,
            'confidence': float(probabilities[predicted.item()]),
            'probabilities': {cmd: float(prob) for cmd, prob in zip(COMMANDS, probabilities)},
            'debug': {
                'avg_level': float(avg_level),
                'max_level': float(np.max(np.abs(audio_data))),
                'samples_processed': len(audio_data)
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_server():
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    print("Загрузка модели...")
    model = load_model()
    print(f"Модель загружена на устройство: {DEVICE}")
    print("Запуск сервера...")
    run_server()