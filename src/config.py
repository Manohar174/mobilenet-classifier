import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Training config
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'mobilenet_classifier.h5')

# Hyperparameters
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 64
EPOCHS = 8
NUM_CLASSES = 3
LEARNING_RATE = 0.001

# Classes mapping (Update these based on your folders)
CLASSES = {0: 'Deer', 1: 'Human', 2: 'Others'}

