import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import json

# ------------------ Settings ------------------
IMG_SIZE = (128, 128)
DATASET_PATH = "dataset/Indian Food Images/"
MODEL_SAVE_PATH = "model/food_cnn_model.h5"
CLASS_INDICES_PATH = "model/class_indices.json"

# ------------------ Data Preprocessing ------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=16,          # smaller batch for faster testing
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# ------------------ Build CNN Model ------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),

    Dense(256, activation="relu"),
    Dropout(0.5),

    Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------------ Callbacks ------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True)
]

# ------------------ Train Model ------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,      # fewer epochs first for faster test
    callbacks=callbacks
)

# ------------------ Save class indices ------------------
with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(train_data.class_indices, f, indent=4)

print(f"✅ Model trained and saved at: {MODEL_SAVE_PATH}")
print(f"✅ Class indices saved at: {CLASS_INDICES_PATH}")
