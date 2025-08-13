import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import pickle

# Load model và class labels
model = load_model("Models/asl_model.keras")
with open('Models/asl_labels.pkl', 'rb') as f:
    class_names = pickle.load(f)

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Mở webcam
cap = cv2.VideoCapture(0)
print("🎥 Camera started. Press Q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Lấy bounding box từ landmarks
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]

            x_min = int(min(x_list) * w) - 80
            y_min = int(min(y_list) * h) - 60
            x_max = int(max(x_list) * w) + 80
            y_max = int(max(y_list) * h) + 40

            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, w)
            y_max = min(y_max, h)

            # Cắt vùng tay
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            # Tiền xử lý ảnh: Gray -> Resize -> (KHÔNG chia 255!)
            hand_gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            hand_resized = cv2.resize(hand_gray, (64, 64))
            hand_input = hand_resized.astype("float32")
            hand_input = np.expand_dims(hand_input, axis=-1)              # (64, 64, 1)
            hand_input = np.expand_dims(hand_input, axis=0) 

            # Dự đoán
            pred = model.predict(hand_input, verbose=0)
            pred_label = class_names[np.argmax(pred)]
            confidence = np.max(pred)

            # Hiển thị
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"{pred_label} ({confidence:.2f})",
                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)

    # Hiển thị webcam
    cv2.imshow("ASL Recognition (CNN + MediaPipe)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
