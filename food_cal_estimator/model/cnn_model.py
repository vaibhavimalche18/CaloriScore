import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0, ResNet50
import numpy as np
import os
import json

class FoodCNN:
    def __init__(self, input_shape=(224, 224, 3), num_classes=101):
        """
        Initialize Food CNN Model
        
        Args:
            input_shape: Input image dimensions (height, width, channels)
            num_classes: Number of food categories
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_simple_cnn(self):
        """Build a simple CNN architecture from scratch"""
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fourth convolutional block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fully connected layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_transfer_learning_model(self, base_model_name='efficientnet'):
        """
        Build a transfer learning model using pre-trained weights
        
        Args:
            base_model_name: 'efficientnet' or 'resnet50'
        """
        if base_model_name == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError("base_model_name must be 'efficientnet' or 'resnet50'")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=base_model.input, outputs=outputs)
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss function"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
        )
        
    def prepare_data_generators(self, train_dir, validation_dir, batch_size=32):
        """
        Prepare data generators for training and validation
        
        Args:
            train_dir: Path to training data directory
            validation_dir: Path to validation data directory
            batch_size: Batch size for training
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        return train_generator, validation_generator
    
    def train(self, train_generator, validation_generator, epochs=50, model_save_path='model/food_cnn_model.h5'):
        """
        Train the CNN model
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of training epochs
            model_save_path: Path to save the best model
        """
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save class indices for later use
        class_indices = train_generator.class_indices
        with open('model/class_indices.json', 'w') as f:
            json.dump(class_indices, f)
        
        print(f"✅ Model training complete! Saved to {model_save_path}")
        return self.history
    
    def predict_image(self, image_path, class_indices_path='model/class_indices.json'):
        """
        Predict food class from image
        
        Args:
            image_path: Path to image file
            class_indices_path: Path to class indices JSON
        """
        # Load class indices
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        
        # Reverse the dictionary to get index -> class name
        idx_to_class = {v: k for k, v in class_indices.items()}
        
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=self.input_shape[:2]
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Predict
        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Get top 5 predictions
        top5_idx = np.argsort(predictions[0])[-5:][::-1]
        top5_predictions = [(idx_to_class[idx], predictions[0][idx]) for idx in top5_idx]
        
        return {
            'predicted_class': idx_to_class[predicted_class_idx],
            'confidence': float(confidence),
            'top5_predictions': top5_predictions
        }
    
    def load_model(self, model_path):
        """Load a saved model"""
        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")
        
    def fine_tune(self, train_generator, validation_generator, epochs=20, unfreeze_layers=50):
        """
        Fine-tune the model by unfreezing some layers
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of fine-tuning epochs
            unfreeze_layers: Number of layers to unfreeze from the end
        """
        # Unfreeze the last N layers
        for layer in self.model.layers[-unfreeze_layers:]:
            layer.trainable = True
        
        # Recompile with lower learning rate
        self.compile_model(learning_rate=1e-5)
        
        # Continue training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
        ]
        
        fine_tune_history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return fine_tune_history


def train_food_cnn(train_dir, val_dir, model_type='transfer', epochs=50):
    """
    Main training function
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        model_type: 'simple' or 'transfer'
        epochs: Number of training epochs
    """
    # Initialize CNN
    food_cnn = FoodCNN(input_shape=(224, 224, 3), num_classes=101)
    
    # Build model
    if model_type == 'simple':
        food_cnn.build_simple_cnn()
    else:
        food_cnn.build_transfer_learning_model(base_model_name='efficientnet')
    
    # Compile model
    food_cnn.compile_model(learning_rate=0.001)
    
    # Print model summary
    print("\n📊 Model Architecture:")
    food_cnn.model.summary()
    
    # Prepare data generators
    print("\n📁 Loading data...")
    train_gen, val_gen = food_cnn.prepare_data_generators(
        train_dir=train_dir,
        validation_dir=val_dir,
        batch_size=32
    )
    
    # Train model
    print("\n🚀 Starting training...")
    history = food_cnn.train(
        train_generator=train_gen,
        validation_generator=val_gen,
        epochs=epochs,
        model_save_path='food_cal_estimator/model/food_cnn_model.h5'
    )
    
    return food_cnn, history


if __name__ == "__main__":
    # Example usage
    TRAIN_DIR = "food_cal_estimator/dataset/train"
    VAL_DIR = "food_cal_estimator/dataset/validation"
    
    # Check if directories exist
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print("❌ Error: Training or validation directory not found!")
        print(f"Expected directories: {TRAIN_DIR} and {VAL_DIR}")
        print("\nPlease organize your data as follows:")
        print("dataset/")
        print("  train/")
        print("    class1/")
        print("      image1.jpg")
        print("      image2.jpg")
        print("    class2/")
        print("      image1.jpg")
        print("  validation/")
        print("    class1/")
        print("    class2/")
    else:
        # Train the model
        model, history = train_food_cnn(
            train_dir=TRAIN_DIR,
            val_dir=VAL_DIR,
            model_type='transfer',  # Use 'simple' for custom CNN
            epochs=50
        )
