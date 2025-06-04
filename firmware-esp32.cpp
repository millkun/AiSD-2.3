#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <driver/adc.h>
#include "ArduinoJson.h"
#include "GyverMAX7219.h"

const char* ssid = "millkun";
const char* password = "assembler";
const char* serverUrl = "http://192.168.25.116:5000/process_audio";
const unsigned long apiTimeout = 5000;

#define SAMPLE_RATE 16000
#define RECORD_DURATION_MS 1500 // Записываю 1.5 секунды
#define BUFFER_SIZE (SAMPLE_RATE * RECORD_DURATION_MS / 1000) // 24000 сэмплов
#define CHUNK_SIZE 2048 // Размер одного чанка для отправки

#define MATRIX_CS_PIN 27
#define MATRIX_DIN_PIN 13
#define MATRIX_CLK_PIN 14
#define MATRIX_NUM 1

#define BUTTON_PIN 27
#define DEBOUNCE_DELAY 50

MAX7219<MATRIX_NUM, MATRIX_NUM, MATRIX_CS_PIN, MATRIX_DIN_PIN, MATRIX_CLK_PIN> matrix;

int16_t audioBuffer[BUFFER_SIZE];
WiFiClient client;
volatile bool buttonPressed = false;
volatile bool isRecording = false;
volatile bool isProcessing = false;
unsigned long recordStartTime = 0;

// Эмоции
const uint8_t error[] PROGMEM = {0x81, 0x42, 0x24, 0x18, 0x18, 0x24, 0x42, 0x81};
const uint8_t record[] PROGMEM = {0x18, 0x24, 0x42, 0x22, 0x04, 0x08, 0x00, 0x08};
const uint8_t upload[] PROGMEM = {0x18, 0x3C, 0x7E, 0xFF, 0x18, 0x18, 0x18, 0x18};
const uint8_t time_B[] PROGMEM = {0x7C, 0x62, 0x62, 0x7C, 0x7C, 0x62, 0x62, 0x7C};
const uint8_t date[] PROGMEM = {0x18, 0x34, 0x24, 0x24, 0x24, 0x3C, 0xC3, 0x81};
const uint8_t algorithms[] PROGMEM = {0x18, 0x24, 0x24, 0x42, 0x7E, 0x42, 0x42, 0x42};
const uint8_t square[] PROGMEM = {0xFF, 0x81, 0x81, 0x81, 0x81, 0x81, 0x81, 0xFF};

// Прототипы функций...
void IRAM_ATTR buttonISR();
void connectWiFi();
String prepareAudioChunkJson(int startIndex, int count);
bool sendAudioToServer();
void handleCommand(String command);

void IRAM_ATTR buttonISR() { // Поскольку я делаю прерывания, то размещаю в быстрой встроенной памяти для исполняемых инструкций
    static unsigned long lastInterruptTime = 0; // а не во флеш-памяти. Вот эта штука работает быстрее, а мне
    unsigned long interruptTime = millis(); // прерывания надо делать правилньо
    
    if (interruptTime - lastInterruptTime > DEBOUNCE_DELAY && !isRecording && !isProcessing) {
        buttonPressed = true;
    }
    lastInterruptTime = interruptTime;
}

void connectWiFi() { 
    if (WiFi.status() == WL_CONNECTED) return;
    
    Serial.println("Подключение к WiFi...");
    WiFi.begin(ssid, password);
    WiFi.setAutoReconnect(true);
    WiFi.persistent(true);
    
    unsigned long startTime = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - startTime < 15000) {
        delay(500);
        Serial.print(".");
    }
    
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("\nОшибка подключения к WiFi!");
    } else {
        Serial.println("\nПодключено к WiFi!");
        Serial.print("IP адрес: ");
        Serial.println(WiFi.localIP());
    }
}

void printImage(const String& imageName) {
    // Маппинг названий на указатели PROGMEM
    struct IconMapping {
        const char* name;
        const uint8_t* data;
    };
    
    static const IconMapping iconMap[] PROGMEM = {
        {"error", error},
        {"record", record},
        {"upload", upload},
        {"time_B", time_B},
        {"date", date},
        {"algorithms", algorithms},
        {"square", square}
    };

    const uint8_t* iconPtr = nullptr;
    for (const auto& mapping : iconMap) {
        if (imageName.equalsIgnoreCase(mapping.name)) {
            iconPtr = (const uint8_t*)pgm_read_ptr(&mapping.data);
            break;
        }
    }

    // Давайте рисовать!
    matrix.clear();
    if (iconPtr) {
        matrix.drawBitmap(0, 0, iconPtr, 8, 8);
    } else {
        matrix.dot(3, 3, 1); // Если не найдено, то просто точка в центре
    }
    matrix.update();
}

void recordAudio() {
    Serial.println("Начало записи...");
    isRecording = true;
    matrix.clear();
    printImage("record");
    matrix.update();
    
    unsigned long sampleInterval = 1000000 / SAMPLE_RATE; // 62.5 мкс для 16 кГц
    unsigned long nextSampleTime = micros();
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        int sample = adc1_get_raw(ADC1_CHANNEL_0);
        audioBuffer[i] = sample - 2048;
        
        while (micros() - nextSampleTime < sampleInterval) {
            delayMicroseconds(10);  // Короткие паузы
        }
        nextSampleTime += sampleInterval;  // Следующий момент времени
    }
    
    Serial.println("Запись завершена");
    isRecording = false;
}

String prepareAudioChunkJson(int startIndex, int count) {
    DynamicJsonDocument doc(16384); // Делаю большой, чтобы был запас. По факту я посчитал будет +- 10-12 КБ
    JsonArray audio = doc.createNestedArray("audio");
    
    for (int i = startIndex; i < startIndex + count; i++) {
        audio.add(audioBuffer[i]);
    }
    
    doc["sample_rate"] = SAMPLE_RATE;
    doc["chunk_index"] = startIndex / CHUNK_SIZE;
    doc["total_chunks"] = (BUFFER_SIZE + CHUNK_SIZE - 1) / CHUNK_SIZE;  // Округление вверх
    
    String jsonData;
    serializeJson(doc, jsonData);
    return jsonData;
}

bool sendAudioToServer() {
    isProcessing = true;
    matrix.clear();
    printImage("upload");
    matrix.update();
    
    bool success = true;
    
    for (int i = 0; i < BUFFER_SIZE; i += CHUNK_SIZE) {
        // Сколько сэмплов осталось в буфере?
        int samplesLeft = BUFFER_SIZE - i;
        int currentChunkSize = (samplesLeft < CHUNK_SIZE) ? samplesLeft : CHUNK_SIZE;
        String chunkJson = prepareAudioChunkJson(i, currentChunkSize);
        
        HTTPClient http;
        http.begin(client, serverUrl);
        http.addHeader("Content-Type", "application/json");
        http.setTimeout(apiTimeout);
        
        Serial.print("Отправка чанка ");
        Serial.print(i / CHUNK_SIZE + 1);
        Serial.print(" из ");
        Serial.println(BUFFER_SIZE / CHUNK_SIZE);
        
        int httpCode = http.POST(chunkJson);
        if (httpCode > 0) {
            String response = http.getString();
            if (i + CHUNK_SIZE >= BUFFER_SIZE) { // Последний чанк. Прим.: Раньше было (i == BUFFER_SIZE - CHUNK_SIZE)
                DynamicJsonDocument doc(512);
                DeserializationError error = deserializeJson(doc, response);
                if (!error && doc.containsKey("command")) {
                    handleCommand(doc["command"].as<String>());
                }
            }
        } else {
            Serial.print("Ошибка HTTP: ");
            Serial.println(httpCode);
            success = false;
            break;
        }
        http.end();
    }
    
    isProcessing = false;
    return success;
}

void handleCommand(String command) {
    Serial.print("Получена команда: ");
    Serial.println(command);
    
    if (command == "время") {
        printImage("time_B");
    } else if (command == "дата") {
        printImage("date");
    } else if (command == "алгоритмы") {
        printImage("algorithms");
    } else if (command == "шум") {
        printImage("square");
    }
    delay(2000);
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\nИнициализация системы...");
    
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), buttonISR, FALLING);
    
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC1_CHANNEL_0, ADC_ATTEN_DB_12);
    
    matrix.begin();
    matrix.setBright(5);
    matrix.setRotation(3);
    matrix.clear();
    
    connectWiFi();
    
    Serial.println("Система готова к работе. Нажмите кнопку для записи.");
}

void loop() {
    if (buttonPressed && !isRecording && !isProcessing) {
        buttonPressed = false;
        
        if (WiFi.status() != WL_CONNECTED) {
            connectWiFi();
            if (WiFi.status() != WL_CONNECTED) {
                matrix.clear();
                printImage("error");
                matrix.update();
                delay(2000);
                return;
            }
        }
      
        recordAudio();
        
        if (!sendAudioToServer()) {
            matrix.clear();
            printImage("error");
            matrix.update();
            delay(2000);
        }
      
        matrix.clear();
        matrix.dot(5, 0, 1);
        matrix.update();
    }
    
    delay(10);
}
