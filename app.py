import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("emotion_model.hdf5")
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

st.title("Emotion Detector")
st.write("Detect emotions from uploaded images (online) or webcam (local only)")

# Mode selection
mode = st.radio("Select mode:", ["Image Upload (Online)", "Webcam (Local Only)"])

# -----------------------
# IMAGE UPLOAD MODE
# -----------------------
if mode == "Image Upload (Online)":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        gray = np.array(img.convert("L"))  # convert to grayscale
        try:
            import cv2
            gray = cv2.resize(gray, (48, 48))
        except ImportError:
            # fallback if cv2 is not available (Streamlit Cloud)
            from PIL import ImageOps
            gray = np.array(ImageOps.fit(img.convert("L"), (48,48)))

        gray = gray / 255.0
        gray = gray.reshape(1, 48, 48, 1)

        # Predict
        prediction = model.predict(gray)
        emotion = emotions[np.argmax(prediction)]
        st.success(f"Predicted Emotion: {emotion}")

# -----------------------
# WEBCAM MODE (LOCAL ONLY)
# -----------------------
elif mode == "Webcam (Local Only)":
    try:
        import cv2
        run = st.checkbox('Start Webcam')
        FRAME_WINDOW = st.image([])

        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        while run:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

            for (x, y, w, h) in faces:
                face = gray_frame[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                face = face / 255.0
                face = face.reshape(1, 48, 48, 1)

                pred = model.predict(face)
                emotion = emotions[np.argmax(pred)]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, emotion, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
    except ImportError:
        st.error("Webcam mode requires OpenCV. This works only locally, not online.")