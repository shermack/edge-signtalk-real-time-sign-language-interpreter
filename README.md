# 🧠 Edge SignTalk: Real-Time Sign Language Interpreter  
Powered by Edge Impulse + TensorFlow Lite (INT8) with Holistic Hand Landmark UI

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Edge%20Impulse-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Model-TensorFlow%20Lite%20(INT8)-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/UI-PyQt5-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Language-Python%203.10-yellow?style=for-the-badge"/>
</p>

---

# 🔥 Overview

**Edge SignTalk** is an on-device real-time American Sign Language (ASL) interpreter built with:

- Edge Impulse (96×96 grayscale INT8 classifier)
- TensorFlow Lite (INT8)
- PyQt5 desktop GUI
- MediaPipe for real-time hand/pose overlays
- A hand-cropping pipeline that ensures clean inputs to the model

Runs **fully offline**, making it ideal for:

- Deaf / hard-of-hearing communication  
- Inclusive classrooms  
- Hospitals and public service desks  
- Remote regions with limited connectivity  

Built for the **Edge Impulse + Microsoft AI Hackathon**, demonstrating how edge AI can empower accessibility.

---

# 🎯 Features

### ✔ Real-time ASL Letter Recognition  
INT8-optimized TensorFlow Lite model exported from Edge Impulse.

### ✔ Clean Cropped-Hand Pipeline  
Extracts a perfect **96×96 grayscale crop** of the hand → sends it to the model.

### ✔ MediaPipe Holistic Overlays  
Shows real-time pose + hand landmarks.

### ✔ PyQt5 Modern UI  
Dark-theme hackathon-friendly UI with:

- Camera dropdown selector (supports Iriun, USB, webcam)
- Start/Stop camera buttons
- Live predicted letter
- Optional sentence builder
- Temporal smoothing (prevents jitter)

### ✔ 100% Offline  
Everything runs locally. Zero cloud calls.

---

# 📂 Repository Structure

```
edge-signtalk-real-time-sign-language-interpreter/
│
├── edge-impulse-sdk/            → Edge Impulse C++ SDK 
├── model-parameters/            → EI project metadata
│
├── tflite-model/
│   ├── model.tflite             → INT8 TFLite classifier
│   ├── labels.txt               → ASL labels (A–Z + space + delete)
│   ├── silent_voice.py          → Full PyQt5 UI (MediaPipe + TFLite)
│   └── run_interpreter.py       → Minimal testing script
│
├── requirements.txt             → Python dependencies
├── LICENSE                      → MIT License
└── README.md                    → (this file)
```

---

# 🛠 Installation

### 1. Clone the repository  
```bash
git clone https://github.com/shermack/edge-signtalk-real-time-sign-language-interpreter.git
cd edge-signtalk-real-time-sign-language-interpreter/tflite-model
```

### 2. Install dependencies  
```bash
pip install -r requirements.txt
```

### 3. Run the full GUI  
```bash
python silent_voice.py
```


# 🧩 How It Works

### 1. Frame capture → hand detection (MediaPipe)
MediaPipe Hands extracts coordinates + bounding box.

### 2. Crop and resize to 96×96 (grayscale)
```python
crop = crop_hand(gray, x, y, w, h)
crop = cv2.resize(crop, (96, 96))
crop = crop.reshape(1, 96, 96, 1)
crop_int8 = (crop / 2 - 128).astype(np.int8)  # quantization scaling
```

### 3. Send to INT8 TFLite model  
Model outputs distribution over **A–Z + SPACE + DELETE**.

### 4. Smooth predictions  
A moving average filter ensures:
- No flicker
- Only stable poses produce letters

### 5. UI overlay  
Skeleton and predicted letter are displayed in real time.

---

# 📊 Model Summary

- **Model type:** Image Classification  
- **Input:** 96×96 grayscale  
- **Output:** 28 classes  
- **Quantization:** INT8  
- **Accuracy:** 95.86%  
- **Dataset:** Custom ASL grayscale dataset + EI test set  

---

# 🧪 Known Limitations & Future Work

### 🔹 Single-hand only  
Two-hand letters require new datasets.

### 🔹 Lighting sensitivity  
Extreme darkness can reduce detection accuracy.

### 🔹 Sentence builder not included  
Was planned but cut due to hackathon timing.

---

# 🚀 Future Enhancements

- Two-hand letters (“H”, “K”, “R”) & sentence builders
- Dynamic signs (e.g., “J” movement)
- Text-to-speech sentence reading
- Edge TPU acceleration (Coral)

---

# 📝 License  
MIT License — free for academic and commercial use.

---

# 👤 Author  
**Shermack Salat**  
Nairobi, Kenya

---

# 🤝 Acknowledgements  

- **Edge Impulse** — training pipeline and model optimization  
- **MediaPipe** — real-time hand detection  
- **TensorFlow Lite** — on-device inference
