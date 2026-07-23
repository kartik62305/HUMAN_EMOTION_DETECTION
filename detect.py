import cv2
import numpy as np
from tensorflow.keras.models import load_model

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
# Load Face Detector
# ==========================
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_classifier.empty():
    print("Error: Haar Cascade file not found!")
    exit()

# ==========================
# Open Webcam
# ==========================
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press Q or ESC to Exit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=7,
        minSize=(100, 100)
    )

    if len(faces) > 0:

        # Select Largest Face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))

        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)

        prediction = model.predict(roi, verbose=0)[0]

        emotion = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Green Box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # Emotion Text
        cv2.putText(
            frame,
            f"{emotion} {confidence:.1f}%",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow("Human Emotion Detection", frame)

    key = cv2.waitKey(1)

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()