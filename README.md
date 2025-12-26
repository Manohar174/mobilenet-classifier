
A lightweight, real-time image classification system designed for edge inference. This project utilizes **Transfer Learning** on the MobileNet architecture to detect specific classes (Deer, Human, Others) from a live video feed, providing audio-visual feedback.

##  Project Overview
This repository implements an end-to-end deep learning pipeline, from data preprocessing to real-time inference. The core model is built upon **MobileNet**, chosen for its depthwise separable convolutions which significantly reduce computational cost (FLOPs) compared to standard CNNs, making it suitable for deployment on embedded devices like Raspberry Pi or Jetson Nano.

### Key Features
*   **Transfer Learning**: Fine-tuned a pre-trained MobileNet (ImageNet weights) with a custom dense classification head.
*   **Real-Time Inference**: Optimized OpenCV pipeline capable of processing webcam streams with low latency.
*   **Data Augmentation**: Robust training pipeline implementing rotation, shifting, and zooming to prevent overfitting on small datasets.
*   **Audio Feedback**: Integrated Text-to-Speech (TTS) for immediate classification alerts.
*   **Modular Codebase**: Refactored from scripts into a structured, object-oriented Python package.

##  Technical Architecture

### Model Architecture
The network consists of two main blocks:
1.  **Feature Extractor**: MobileNet (frozen base layers) to extract high-level spatial features.
2.  **Classifier Head**: 
    *   `GlobalAveragePooling2D`: Reduces spatial dimensions ($7 \times 7 \times 1024 \rightarrow 1 \times 1 \times 1024$).
    *   `Dense (64 units, ReLU)`: Intermediate learning layer.
    *   `Dropout (0.5)`: Regularization to prevent overfitting.
    *   `Dense (3 units, Softmax)`: Final probability distribution for 3 classes.

### Preprocessing
*   **Input Size**: $224 \times 224 \times 3$
*   **Normalization**: Inputs are scaled to $[-1, 1]$ interval using the specific MobileNet preprocessing function, ensuring distribution matches the ImageNet pre-training.

##  Directory Structure
mobilenet-classifier/
│
├── data/ # Dataset directory (gitignored)
│ ├── train/ # Training images organized by class
│ └── valid/ # Validation images
├── models/ # Serialized model files (.h5)
├── src/ # Source code package
│ ├── init.py
│ ├── config.py # Centralized configuration & hyperparameters
│ ├── train.py # Training pipeline with DataGenerators
│ └── inference.py # Inference class for decoupling logic
├── run_webcam.py # Main execution script
├── requirements.txt # Project dependencies
└── README.md # Documentation


##  Getting Started

### Prerequisites
*   Python 3.8+
*   Webcam for real-time demo

### Installation
1.  **Clone the repository**
    ```
    git clone https://github.com/Manohar174/mobilenet-classifier.git
    cd mobilenet-classifier
    ```

2.  **Install dependencies**
    It is recommended to use a virtual environment.
    ```
    pip install -r requirements.txt
    ```

### Training the Model
To train on your own dataset, organize your images in `data/train` and `data/valid` folders, then run:
python -m src.train

text

### Running Inference
Start the real-time webcam classifier:
python run_webcam.py

text
*   Press `ESC` to exit the application.

##  Parameters
*   **Optimizer**: Adam 
*   **Loss Function**: Categorical Crossentropy
*   **Input Resolution**: 224x224 px

*(Note: Add specific accuracy/loss metrics here after you finish training, e.g., "Achieved 92% validation accuracy after 100 epochs.")*

##  Future Improvements
*   **ROS2 Integration**: Wrap the inference logic into a ROS2 Node for autonomous robot navigation.
*   **Quantization**: Convert the model to TensorFlow Lite (TFLite) for further speedup on edge hardware.
*   **Object Detection**: Transition from classification to detection (e.g., SSD-MobileNet) to localize objects in the frame.

##  License
This project is licensed under the MIT License.
