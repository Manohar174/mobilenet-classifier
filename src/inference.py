import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from src import config
import cv2

class Classifier:
    def __init__(self, model_path=config.MODEL_SAVE_PATH):
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path)
        self.classes = config.CLASSES

    def predict_frame(self, frame):
        # Resize frame
        img_resized = cv2.resize(frame, (config.IMG_WIDTH, config.IMG_HEIGHT))
        
        # Expand dimensions to create batch (1, 224, 224, 3)
        img_array = np.expand_dims(img_resized, axis=0)
        
        # Preprocess input (CRITICAL: Must match training)
        # MobileNet preprocess_input scales to [-1, 1]
        # Your previous code did /255.0 which was incorrect for this model
        img_preprocessed = preprocess_input(img_array.astype(np.float32))
        
        prediction = self.model.predict(img_preprocessed, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return self.classes.get(class_idx, "Unknown"), confidence

