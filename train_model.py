
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import os

# Paths & Labels
data_dir = 'wm811k/Maps'  # Adjust this path based on your dataset
classes = os.listdir(data_dir)
df = []
for cls in classes:
    for img in os.listdir(f'{data_dir}/{cls}'):
        label = 0 if cls == 'none' else 1
        df.append([f'{data_dir}/{cls}/{img}', label])
df = pd.DataFrame(df, columns=['filepath','label'])

# Train/Val split
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)

# Data generators
IMG_SIZE = (128,128)
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=20)
val_datagen = ImageDataGenerator(rescale=1./255)
train_gen = train_datagen.flow_from_dataframe(train_df, x_col='filepath', y_col='label',
    target_size=IMG_SIZE, class_mode='binary', batch_size=32)
val_gen = val_datagen.flow_from_dataframe(val_df, x_col='filepath', y_col='label',
    target_size=IMG_SIZE, class_mode='binary', batch_size=32)

# CNN model
model = models.Sequential([
    layers.Conv2D(32,3,activation='relu',input_shape=(*IMG_SIZE,3)),
    layers.MaxPooling2D(), layers.Conv2D(64,3,activation='relu'),
    layers.MaxPooling2D(), layers.Conv2D(128,3,activation='relu'),
    layers.MaxPooling2D(), layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=10)
model.save('wafer_pass_fail.h5')
