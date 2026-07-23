import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# ==========================
# Page Settings
# ==========================
st.set_page_config(
    page_title="Human Emotion Detection",
    layout="centered"
)

st.title("😊 Human Emotion Detection using CNN")

# ==========================
# Load Model
# ==========================
model = load_model("model/emotion_model.h5")

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# ==========================
# Load Haar Cascade
# ==========================
cascade_path = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    st.error(f"Haar Cascade file not found!\nPath: {cascade_path}")
    st.stop()

# ==========================
# Camera
# ==========================
camera = st.camera_input("📷 Take a Photo")

if camera is not None:

    file_bytes = np.asarray(bytearray(camera.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) > 0:

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)

        prediction = model.predict(roi, verbose=0)[0]

        emotion = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(
            img,
            f"{emotion} {confidence:.1f}%",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        st.success(f"Emotion: {emotion} ({confidence:.1f}%)")

    else:
        st.warning("No Face Detected")

    st.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        use_container_width=True
    )