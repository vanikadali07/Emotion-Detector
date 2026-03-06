# Emotion Detection from Face using Deep Learning

## 📌 Project Overview

This project detects **human emotions from facial expressions** using **Deep Learning and Computer Vision**.
The system captures a face using a webcam and predicts the emotion in real time.

The model is trained on a facial emotion dataset and can classify emotions such as:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

This project demonstrates the use of **CNN (Convolutional Neural Networks)** for image classification.

---

## 🚀 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib

---

## 🧠 Model Architecture

The project uses a **Convolutional Neural Network (CNN)** for emotion classification.

Steps involved:

1. Face detection using **Haar Cascade (OpenCV)**
2. Face preprocessing (grayscale, resizing)
3. Emotion prediction using **trained CNN model**

---

## 📊 Dataset

The model is trained using the **FER2013 facial emotion dataset** which contains thousands of labeled facial expression images.

Dataset contains emotions like:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

---

## ⚙️ Installation

Clone the repository:

git clone https://github.com/vanikadali07/Emotion-Detector.git

Navigate to the project folder:

cd Emotion-Detector

Install dependencies:

pip install -r requirements.txt

---

## ▶️ How to Run the Project

### 1️⃣ Train the Model

python train_model.py

This will generate the trained model file:

emotion_model.hdf5

### 2️⃣ Run Emotion Detection

python detect_emotion.py

This will open your **webcam** and detect emotions in real time.

---

## 📸 Output

The system:

* Detects faces using OpenCV
* Predicts emotions using the trained CNN model
* Displays emotion label on the detected face

---

## 🎯 Applications

* Human Computer Interaction
* Mental health monitoring
* Smart surveillance systems
* Customer behavior analysis
* Interactive AI systems

---

## 👩‍💻 Author

**Vani Kadali**
B.Tech CSE (AI & ML)

GitHub:
https://github.com/vanikadali07

---

## ⭐ If you like this project

Please consider **starring the repository**.
