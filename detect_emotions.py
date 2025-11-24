# detect_emotion.py - Real-time Emotion Detection
import cv2
import numpy as np
import tensorflow as tf

print("🎭 REAL-TIME FACE EMOTION DETECTION")
print("Loading trained AI model...")

try:
    # Load trained model
    model = tf.keras.models.load_model('model/high_accuracy_emotion_model.h5')
    emotions = ['😠 Angry', '😊 Happy', '😢 Sad', '😲 Surprise']
    print("✅ AI model loaded successfully!")
except:
    print("❌ Model not found. Please run train_ai.py first!")
    exit()

print("Starting webcam...")

# Initialize webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("🔴 Press 'Q' to quit detection")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Extract and preprocess face
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi.reshape(1, 48, 48, 1).astype(np.float32) / 255.0
        
        # Predict emotion
        prediction = model.predict(face_roi, verbose=0)
        emotion_idx = np.argmax(prediction)
        confidence = prediction[0][emotion_idx]
        
        # Display results
        emotion_text = f"{emotions[emotion_idx]} ({confidence:.1%})"
        color = (0, 255, 0) if confidence > 0.6 else (0, 165, 255)  # Green if confident
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, emotion_text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.imshow('Face Emotion AI - Press Q to quit', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Emotion detection completed!")