from flask import Flask, request, jsonify
import numpy as np
import os
import time
import torch
import torch.nn as nn
from torchvision.models import resnet18
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle


from fill_data import * # Функции создания спектрограммы и нормализации в fill_data.py

app = Flask(__name__)

# Настройки
SAMPLE_RATE = 16000
AUDIO_FOLDER = "raw_audio"
SPECTROGRAM_FOLDER = "spectrograms"
BUFFER_SIZE = 24000  # 24000 сэмплов (= 1.5 сек)
CHUNK_SIZE = 2048    # Как в прошивке МК размер чанка
EXPECTED_CHUNKS = (BUFFER_SIZE + CHUNK_SIZE - 1) // CHUNK_SIZE # 12 чанков (24000/2048)

os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)

# Тут класс модели, такой же как при обучении
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
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Загружаю модель
def load_model(model_path, num_classes=3):
    model = ResNetSpectrogram(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Тут подготовка данных
def prepare_input(audio):
    # Делаю float, потому что с МК идет int16_t, а обучали на флоат
    audio = audio.astype(np.float32)
    audio = librosa.util.normalize(audio)
    
    mel_spec = create_mel_spec(audio) # Основные действия
    input_data = normalize(mel_spec)
    
    # Добавляю размерности для модели (1, 1, H, W)
    input_tensor = torch.FloatTensor(input_data).unsqueeze(0).unsqueeze(0)
    
    return input_tensor, mel_spec

# Сохраняю спектрограммы как изображения, чтобы проверять работоспособность
def save_spectrogram_image(spec, filename):
    try:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spec, sr=SAMPLE_RATE, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()  # Явное закрытие фигуры
    except Exception as e:
        print(f"Error saving spectrogram: {e}")

# Классы команд
CLASS_NAMES = ["дата", "алгоритмы", "время", "шум"]

# Загружаю лучшую модель
model = load_model("best_model.pth", num_classes=len(CLASS_NAMES))

# Глобальный буфер здесь
current_session = {
    "id": None,
    "chunks": [],
    "total_samples": 0
}

@app.route('/process_audio', methods=['POST'])
def process_audio():
    global current_session

    try:
        data = request.get_json()
        
        if 'audio' not in data:
            return jsonify({"error": "No audio data"}), 400
        
        chunk_index = data["chunk_index"]
        total_chunks = data["total_chunks"]
        audio_chunk = np.array(data["audio"], dtype=np.int16)
        
        # Если это первый чанк — начинаю новую сессию
        if chunk_index == 0:
            current_session["id"] = int(time.time())
            current_session["chunks"] = []
            current_session["total_samples"] = 0
            print(f"\n🔥 Начата новая сессия {current_session['id']}")

        # Проверяю, что чанк пришел в правильной сессии
        if len(current_session["chunks"]) != chunk_index:
            print(f"⛔️ Ошибка порядка чанков! Ожидался {len(current_session['chunks'])}, получен {chunk_index}")
            return jsonify({"error": "Invalid chunk order"}), 400

        # Добавляю чанк в буфер
        current_session["chunks"].append(audio_chunk)
        current_session["total_samples"] += len(audio_chunk)
        
        print(f"✅ Чанк {chunk_index + 1}/{total_chunks} принят ({len(audio_chunk)} сэмплов)")

        # Если это последний чанк — обрабатываем всё
        if chunk_index == total_chunks - 1:
            if current_session["total_samples"] >= BUFFER_SIZE - CHUNK_SIZE:  # Последний чанк
                if current_session["total_samples"] != BUFFER_SIZE:
                    print(f"⛔️ Ошибка! Получено {current_session['total_samples']}, ожидалось {BUFFER_SIZE}")
                    return jsonify({"error": "Incomplete audio data"}), 400

            # Склеиваем все чанки в один массив
            full_audio = np.concatenate(current_session["chunks"])
            
            # Сохраняю сырые аудио данные, опять же для отладки потом программы
            audio_filename = f"{AUDIO_FOLDER}/audio_{current_session['id']}.pkl"
            with open(audio_filename, 'wb') as f:
                pickle.dump(full_audio, f)
            
            # Подготовка входных данных для модели
            input_tensor, mel_spec = prepare_input(full_audio)
            
            # Сохраняю картинку спектрограммки
            spec_image_filename = f"{SPECTROGRAM_FOLDER}/spec_{current_session['id']}.png"
            save_spectrogram_image(mel_spec, spec_image_filename)
            
            # Гадаем в гадалки, предсказываем предсказалки
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                command = CLASS_NAMES[predicted.item()]
            
            print(f"\n💾 Аудио сохранено в {audio_filename}")
            print(f"📊 Спектрограмма сохранена в {spec_image_filename}")
            print(f"🤖 Распознанная команда: {command}")
            
            return jsonify({
                "status": "success",
                "samples_received": full_audio.size,
                "command": command
            })
        else:
            return jsonify({"status": "chunk_received"})

    except Exception as e:
        print(f"⛔️ Ошибка: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"🚀 Сервер запущен. Ожидается {EXPECTED_CHUNKS} чанков по {CHUNK_SIZE} сэмплов")
    print(f"🔊 Классы команд: {CLASS_NAMES}")
    app.run(host='0.0.0.0', port=5000, debug=True)
