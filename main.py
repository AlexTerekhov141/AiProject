import os
import numpy as np
import pandas as pd
import librosa
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
warnings.filterwarnings("ignore")

# Путь к папке с данными
dataset_path = "dataset"

# Эмоции в RAVDESS
emotion_labels = {
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
    5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'
}

# Функция для извлечения признаков (MFCC)
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')  # Загрузка аудио
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)       # Извлечение MFCC
    mfcc_scaled = np.mean(mfcc.T, axis=0)                                # Усреднение по времени
    return mfcc_scaled

# Создание списка для хранения данных
features = []
emotions = []

# Проход по всем папкам Actor_01, Actor_02, ..., Actor_24
for actor_folder in os.listdir(dataset_path):
    actor_path = os.path.join(dataset_path, actor_folder)
    if os.path.isdir(actor_path):  # Проверяем, что это папка
        for file in os.listdir(actor_path):
            file_path = os.path.join(actor_path, file)
            if file_path.endswith(".wav"):  # Проверяем, что это аудиофайл
                # Извлекаем эмоцию из имени файла
                part = file.split("-")
                emotion = int(part[2])  # Эмоция закодирована в 3-й части имени
                features.append(extract_features(file_path))
                emotions.append(emotion_labels[emotion])

# Преобразование в массивы NumPy
X = np.array(features)
y = np.array(emotions)

# Вывод информации о данных
print("Признаки X:", X.shape)
print("Метки y:", y.shape)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Разделение на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Построение модели
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(len(encoder.classes_), activation='softmax')  # Количество эмоций
])

# Компиляция модели
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Оценка модели
loss, accuracy = model.evaluate(X_test, y_test)
print("Точность на тестовом наборе:", accuracy)
def predict_emotion(file_path, model, encoder):
    feature = extract_features(file_path).reshape(1, -1)  # Извлечение признаков
    prediction = model.predict(feature)
    predicted_label = np.argmax(prediction, axis=1)
    emotion = encoder.inverse_transform(predicted_label)
    return emotion[0]
def predict_emotion_percentages(file_path, model, encoder):
    feature = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(feature)
    percentages = prediction[0] * 100
    emotion_probabilities = {emotion: percentage for emotion, percentage in zip(encoder.classes_, percentages)}
    return emotion_probabilities

file_path = "Potter.wav"
file_path2 = "Onegin.wav"
predicted_emotion = predict_emotion(file_path, model, encoder)
predicted_emotion2 = predict_emotion(file_path2, model, encoder)
print("Предсказанная эмоция:", predicted_emotion)
print("Предсказанная эмоция:", predicted_emotion2)
file_path = "Potter.wav"
predicted_percentages = predict_emotion_percentages(file_path, model, encoder)

print("Предсказанные вероятности эмоций:")
for emotion, percentage in predicted_percentages.items():
    print(f"{emotion}: {percentage:.2f}%")