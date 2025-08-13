import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load model và class labels
model = load_model("asl_model.keras")
with open('asl_labels.pkl', 'rb') as f:
    class_names = pickle.load(f)

print("✅ Number of classes:", len(class_names))

# Load ảnh đầu vào
img = cv2.imread("G.jpg")

# Tiền xử lý ảnh: resize + reshape (KHÔNG chia 255)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (64, 64))
img = img.astype("float32")                     # ❗ KHÔNG chia /255.0
img = np.expand_dims(img, axis=-1)              # (64, 64, 1)
img = np.expand_dims(img, axis=0)               # (1, 64, 64, 1)

# Dự đoán
pred = model.predict(img, verbose=0)
label = class_names[np.argmax(pred)]
confidence = np.max(pred)

print(f"🔍 Prediction: {label} ({confidence:.2f})")
