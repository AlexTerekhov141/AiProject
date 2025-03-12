import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model

model = load_model('my_model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)


def extract_features(file_path, n_mfcc=20):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled


def predict_audio(file_path):
    features = extract_features(file_path, n_mfcc=20)
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_scaled = np.expand_dims(features_scaled, axis=2)
    prediction = model.predict(features_scaled)
    predicted_label = encoder.inverse_transform(prediction)

    return predicted_label[0][0]


if __name__ == "__main__":
    audio_file = 'audio_speech_actors_01-24/Actor_03/03-01-05-02-01-02-03.wav'
    predicted = predict_audio(audio_file)
    print("Предсказанная метка:", predicted)
