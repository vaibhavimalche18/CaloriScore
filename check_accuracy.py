import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os
import shutil

MODEL_PATH = "model/food_cnn_model.h5"
CLASS_INDICES_PATH = "model/class_indices.json"
DATASET_PATH = "dataset/Indian Food Images/"
FILTERED_DATASET_PATH = "dataset/Indian Food Images_filtered/"
IMG_SIZE = (128, 128)

# ---------------------------------------
# Load model
# ---------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------------------------------
# Load class indices (20 classes)
# ---------------------------------------
with open(CLASS_INDICES_PATH, "r") as f:
    model_classes = list(json.load(f).keys())

print("Model was trained on:", model_classes)
print(f"Total Model Classes: {len(model_classes)}")


# ---------------------------------------
# Filter dataset to match model classes
# ---------------------------------------
if os.path.exists(FILTERED_DATASET_PATH):
    shutil.rmtree(FILTERED_DATASET_PATH)

os.makedirs(FILTERED_DATASET_PATH, exist_ok=True)

for class_name in model_classes:
    src = os.path.join(DATASET_PATH, class_name)
    dst = os.path.join(FILTERED_DATASET_PATH, class_name)

    if os.path.exists(src):
        shutil.copytree(src, dst)
    else:
        print(f"⚠ WARNING: Folder not found in dataset: {class_name}")

print("✅ Filtered dataset created successfully.\n")


# ---------------------------------------
# Load filtered training data
# ---------------------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    FILTERED_DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# ---------------------------------------
# Evaluate model on TRAINING dataset
# ---------------------------------------
loss, acc = model.evaluate(train_data)

print("\n=============== TRAINING ACCURACY ===============")
print(f"Training Loss: {loss:.4f}")
print(f"Training Accuracy: {acc * 100:.2f}%")
print("=================================================\n")
