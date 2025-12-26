import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.optimizers import Adam
import math
import os
from src import config

def build_model():
    base_model = MobileNet(include_top=False, input_shape=(config.IMG_WIDTH, config.IMG_HEIGHT, 3))
    
    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False
        
    inputs = Input(shape=(config.IMG_WIDTH, config.IMG_HEIGHT, 3))
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(config.NUM_CLASSES, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)

def train():
    # Ensure model directory exists
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # Data Generators
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, # Critical: Scales to [-1, 1]
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2
    )
    
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(config.IMG_WIDTH, config.IMG_HEIGHT),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_directory(
        config.VALID_DIR,
        target_size=(config.IMG_WIDTH, config.IMG_HEIGHT),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        class_mode='categorical'
    )

    model = build_model()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        metrics=['accuracy']
    )

    print("Starting training...")
    model.fit(
        train_generator,
        steps_per_epoch=math.ceil(train_generator.samples / config.BATCH_SIZE),
        epochs=config.EPOCHS,
        validation_data=validation_generator,
        validation_steps=math.ceil(validation_generator.samples / config.BATCH_SIZE)
    )
    
    model.save(config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()

