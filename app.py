import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
model = load_model("emotion_model.hdf5")  # make sure your model file is in the same folder
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

st.title("Emotion Detector")
st.write("Detect emotions from images or webcam")

# Mode selection
mode = st.radio("Select mode:", ["Image Upload (Online Demo)", "Webcam (Local Only)"])

# -------------------------------
# IMAGE UPLOAD MODE (works online)
# -------------------------------
if mode == "Image Upload (Online Demo)":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess for model
        img = np.array(img.convert("L"))
        img = cv2.resize(img, (48,48))
        img = img / 255.0
        img = img.reshape(1,48,48,1)

        prediction = model.predict(img)
        emotion = emotions[np.argmax(prediction)]
        st.success(f"Predicted Emotion: {emotion}")

# -------------------------------
# WEBCAM MODE (LOCAL ONLY)
# -------------------------------
elif mode == "Webcam (Local Only)":
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48,48))
            face = face / 255.0
            face = face.reshape(1,48,48,1)

            pred = model.predict(face)
            emotion = emotions[np.argmax(pred)]

            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()