import zipfile
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedShuffleSplit

def load_images_from_zip(zip_path):
    images = []
    labels = []
    with zipfile.ZipFile(zip_path, 'r') as archive:
        for file in archive.namelist():
            if file.startswith('Car-Bike-Dataset/Bike') and (file.endswith('.jpg') or file.endswith('.jpeg')):
                label = 0
            elif file.startswith('Car-Bike-Dataset/Car') and (file.endswith('.jpg') or file.endswith('.jpeg')):
                label = 1
            else:
                continue
            with archive.open(file) as file_handle:
                image = Image.open(file_handle).convert('RGB')
                image = image.resize((128, 128))
                image_array = np.array(image) / 255.0
                images.append(image_array)
                labels.append(label)
    return np.array(images), np.array(labels)

X, Y = load_images_from_zip('cars_bikes_data.zip')

sss = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, test_idx = next(sss.split(X, Y))
x_train, x_test = X[train_idx], X[test_idx]
y_train, y_test = Y[train_idx].astype(np.float32), Y[test_idx].astype(np.float32)

model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(32, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16,
          epochs=30,
          validation_data=(x_test, y_test))

predictions = model.predict(x_test)
binary_predictions = (predictions > 0.5).astype(int).flatten()
accuracy = np.mean(y_test == binary_predictions)
print(f'Точность предсказания на тестовых данных : {accuracy * 100:.5f}%')
