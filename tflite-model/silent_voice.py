import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QMainWindow
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer
from tensorflow.lite.python.interpreter import Interpreter

# ==========================================================
# CONFIG
# ==========================================================
MODEL_PATH = "model.tflite"
LABELS_PATH = "labels.txt"

IMG_SIZE = 96
CONF_THRESHOLD = 0.75
STABILITY_FRAMES = 6          # letter must stay consistent this many frames
SMOOTH_WINDOW = 8             # smoothing predictions

# ==========================================================
# LOAD LABELS
# ==========================================================
with open(LABELS_PATH, "r") as f:
    LABELS = [line.strip() for line in f.readlines()]

# ==========================================================
# LOAD TFLITE MODEL
# ==========================================================
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_SHAPE = input_details[0]["shape"]
INPUT_DTYPE = input_details[0]["dtype"]
OUTPUT_DTYPE = output_details[0]["dtype"]

print("Model input:", input_details)
print("Model output:", output_details)

# ==========================================================
# GRAYSCALE + CONTRAST ENHANCE PIPELINE
# ==========================================================
def preprocess_crop(crop):
    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # CLAHE contrast enhancement (best for grayscale datasets)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Resize to model size
    resized = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))

    # Normalize → INT8 quantization
    img = resized.astype(np.float32) / 255.0
    img = img - 0.5
    img = img * 2.0

    img = np.expand_dims(img, axis=-1)          # (96,96,1)
    img = np.expand_dims(img, axis=0)           # (1,96,96,1)

    # Convert to int8 for model
    img = (img / input_details[0]["quantization"][0] + 
           input_details[0]["quantization"][1]).astype(np.int8)

    return img

# ==========================================================
# HAND DETECTION (MediaPipe Hands)
# ==========================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================================
# MAIN APP
# ==========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Edge SignTalk – Silent Voice")
        self.setFixedSize(1280, 750)
        self.setStyleSheet("background-color: black; color: white;")

        # UI
        self.video_label = QLabel()
        self.video_label.setFixedSize(820, 540)
        self.video_label.setStyleSheet("background-color: #101010; border: 2px solid #444;")

        self.letter_box = QTextEdit()
        self.letter_box.setFixedHeight(90)
        self.letter_box.setReadOnly(True)
        self.letter_box.setFont(QFont("Segoe UI", 28))

        # Dropdown camera list
        self.cam_box = QComboBox()
        self.populate_cameras()

        # Buttons
        self.start_btn = QPushButton("Start Camera")
        self.start_btn.setStyleSheet("background-color: #00CC66; font-size: 16px;")
        self.start_btn.clicked.connect(self.start_camera)

        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.setStyleSheet("background-color: #CC0033; font-size: 16px;")
        self.stop_btn.clicked.connect(self.stop_camera)

        # Layout
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Predicted Letter:", font=QFont("Segoe UI", 20)))
        left_layout.addWidget(self.letter_box)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.video_label)
        right_layout.addWidget(self.cam_box)
        right_layout.addWidget(self.start_btn)
        right_layout.addWidget(self.stop_btn)

        main = QHBoxLayout()
        main.addLayout(left_layout, 1)
        main.addLayout(right_layout, 2)

        container = QWidget()
        container.setLayout(main)
        self.setCentralWidget(container)

        # Camera & prediction buffers
        self.cap = None
        self.pred_history = []
        self.last_final_letter = ""

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    # -----------------------------------------------
    def populate_cameras(self):
        for i in range(8):
            cap = cv2.VideoCapture(i)
            ok, _ = cap.read()
            if ok:
                self.cam_box.addItem(f"Camera {i}", i)
            cap.release()

    # -----------------------------------------------
    def start_camera(self):
        idx = self.cam_box.currentData()
        self.cap = cv2.VideoCapture(idx)
        self.timer.start(30)

    # -----------------------------------------------
    def stop_camera(self):
        if self.cap:
            self.cap.release()
        self.timer.stop()

    # -----------------------------------------------
    def update_frame(self):
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        hand_crop = None
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            h, w, _ = frame.shape

            # Get bounding box
            xs = [lm.x for lm in hand.landmark]
            ys = [lm.y for lm in hand.landmark]

            min_x = int(min(xs) * w) - 25
            max_x = int(max(xs) * w) + 25
            min_y = int(min(ys) * h) - 25
            max_y = int(max(ys) * h) + 25

            min_x = max(min_x, 0)
            min_y = max(min_y, 0)
            max_x = min(max_x, w)
            max_y = min(max_y, h)

            hand_crop = frame[min_y:max_y, min_x:max_x]

            # Draw landmarks onscreen for demo
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

        # Predict if crop exists
        if hand_crop is not None:
            img = preprocess_crop(hand_crop)

            interpreter.set_tensor(input_details[0]["index"], img)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]["index"])[0]

            # Dequantize
            scale, zero = output_details[0]["quantization"]
            preds = (out.astype(np.float32) - zero) * scale

            idx = int(np.argmax(preds))
            conf = float(np.max(preds))
            letter = LABELS[idx]

            # Smoothing
            self.pred_history.append(letter)
            if len(self.pred_history) > SMOOTH_WINDOW:
                self.pred_history.pop(0)

            stable_letter = max(set(self.pred_history), key=self.pred_history.count)
            stability = self.pred_history.count(stable_letter)

            # Only output when stable & confident
            if stability >= STABILITY_FRAMES and conf >= CONF_THRESHOLD:
                if stable_letter != self.last_final_letter:
                    self.letter_box.setText(stable_letter)
                    self.last_final_letter = stable_letter

        # Display video
        disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = disp.shape
        img_qt = QImage(disp.data, w, h, w * ch, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img_qt))


# ==========================================================
# RUN APP
# ==========================================================
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())