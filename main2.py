import numpy as np
import pandas as pd
import librosa
import warnings
import os
import keras
import matplotlib.pyplot as plt
import seaborn as sns

from keras.src.layers import Conv1D, MaxPooling1D, LSTM
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import pickle

csv = 'features.csv'
Features = pd.read_csv(csv)
X = Features.iloc[: ,:-1].values
Y = Features['labels'].values
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=24, shuffle=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

model = Sequential()
model.add(Conv1D(2048, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2, strides = 2, padding = 'same'))
model.add(BatchNormalization())

model.add(Conv1D(1024, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides = 2, padding = 'same'))
model.add(BatchNormalization())

model.add(Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides = 2, padding = 'same'))
model.add(BatchNormalization())

model.add(LSTM(256, return_sequences=True))

model.add(LSTM(128))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(16, activation='softmax'))

optimiser = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

print(model.summary())
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("Объекты scaler и encoder успешно сохранены!")
history = model.fit(x_train, y_train, batch_size=64, epochs=200, validation_data=(x_test, y_test))

print("Accuracy of our model on test data : " , model.evaluate(x_test,y_test)[1]*100 , "%")

model.save('my_model.h5')

pred_test = model.predict(x_test)
y_pred = encoder.inverse_transform(pred_test)

y_test = encoder.inverse_transform(y_test)

df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df['Predicted Labels'] = y_pred.flatten()
df['Actual Labels'] = y_test.flatten()

df.head(20)


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
audio_file = 'audio_speech_actors_01-24/Actor_03/03-01-05-02-01-02-03.wav'
predicted = predict_audio(audio_file)
print("Предсказанная метка:", predicted)

