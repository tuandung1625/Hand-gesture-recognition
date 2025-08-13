import cv2
import numpy as np
import tensorflow as tf
import pickle

# Load model
model = tf.keras.models.load_model("asl_model.keras")
with open("asl_labels.pkl", "rb") as f:
    class_names = pickle.load(f)

IMG_SIZE = (64, 64)

def preprocess_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    norm = resized.astype('float32') / 255.0
    norm = np.expand_dims(norm, axis=-1)  # (64, 64, 1)
    norm = np.expand_dims(norm, axis=0)   # (1, 64, 64, 1)
    return norm

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Lật gương

    # Vẽ vùng ROI ở giữa
    h, w = frame.shape[:2]
    roi_size = 224
    x1 = w // 2 - roi_size // 2
    y1 = h // 2 - roi_size // 2
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    # Vẽ khung xanh cho ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Cắt vùng ROI
    roi = frame[y1:y2, x1:x2]
    if roi.shape[0] != roi_size or roi.shape[1] != roi_size:
        continue

    # Predict
    img_input = preprocess_roi(roi)
    preds = model.predict(img_input, verbose=0)
    pred_class = np.argmax(preds)
    confidence = np.max(preds)

    label = f"{class_names[pred_class]} ({confidence:.2f})"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # Hiển thị
    cv2.imshow("ASL Recognition (press Q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()