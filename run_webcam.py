import cv2
import pyttsx3
import threading
from src.inference import Classifier

def speak(engine, text):
    """Helper to run speech in a separate loop to avoid freezing video"""
    if not engine._inLoop:
        engine.say(text)
        engine.runAndWait()

def main():
    # Initialize Classifier
    classifier = Classifier()
    
    # Initialize TTS
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    print("Starting Webcam... Press 'ESC' to exit.")

    frame_count = 0
    current_label = "Initializing..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optimization: Run prediction only every 5 frames to reduce lag
        if frame_count % 5 == 0:
            label, confidence = classifier.predict_frame(frame)
            current_label = f"{label} ({confidence:.2f})"
            
            # Simple logic to speak only if confidence is high
            if confidence > 0.8:
                # Use a non-blocking approach or a simple print to avoid lag
                # engine.say(label)  # Uncommenting might cause lag depending on system
                pass

        # Display text
        cv2.putText(frame, f"Pred: {current_label}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Real-time Classifier', frame)

        if cv2.waitKey(1) & 0xFF == 27: # ESC key
            break
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

