🧠 Edge SignTalk: Real-Time Sign Language Interpreter

Powered by Edge Impulse + TensorFlow Lite (INT8) with Holistic Hand Landmark UI

<p align="center"> <img src="https://img.shields.io/badge/Platform-Edge%20Impulse-blue?style=for-the-badge"/> <img src="https://img.shields.io/badge/Model-TensorFlow%20Lite%20(INT8)-green?style=for-the-badge"/> <img src="https://img.shields.io/badge/UI-PyQt5-purple?style=for-the-badge"/> <img src="https://img.shields.io/badge/Language-Python%203.10-yellow?style=for-the-badge"/> </p>
🔥 Overview

Edge SignTalk is an on-device real-time American Sign Language (ASL) interpreter built with:

Edge Impulse (image-based 96×96 INT8 classifier)

TensorFlow Lite (INT8)

PyQt5 desktop interface

MediaPipe skeleton/hand overlays for visual clarity

Camera cropping pipeline ensuring clean hand-only inputs

Designed for speed, privacy, and offline use, Edge SignTalk runs fully on-device without internet, making it suitable for:

Deaf / hard-of-hearing communication

Inclusive classrooms

Hospitals & public service desks

Remote regions with limited connectivity

This submission was built for the Edge Impulse + Microsoft AI Hackathon to demonstrate how edge AI can empower accessibility in the real world.

🎯 Features
✔ Real-time ASL Letter Recognition

INT8-optimized TensorFlow Lite model (exported from Edge Impulse).

✔ Clean Cropped-Hand Pipeline

Instead of full-frame guessing, the system extracts a perfect 96×96 hand crop before classification.

✔ MediaPipe Holistic Overlays

Shows hand/pose landmarks on screen for transparency and user trust.

✔ PyQt5 Modern UI

Black-themed, hackathon-ready interface with:

Camera dropdown selector (supports Iriun, USB, webcam)

Start/stop camera button

Real-time predicted letter display

Sentence builder (optional extension)

Smooth prediction with temporal filtering

✔ Runs Fully Offline

All inference is performed locally.

📂 Repository Structure
edge-signtalk-real-time-sign-language-interpreter/
│
├── edge-impulse-sdk/            → Edge Impulse C++ SDK (auto-generated)
├── model-parameters/            → Metadata from Edge Impulse
│
├── tflite-model/
│   ├── model.tflite             → INT8 model (main classifier)
│   ├── labels.txt               → 28 ASL classes
│   ├── silent_voice.py          → Full PyQt5 UI with MediaPipe + TFLite
│   └── run_interpreter.py       → Minimal model test script
│
├── requirements.txt             → Python dependencies
├── LICENSE                      → MIT License
└── README.md                    → (You are reading this)

🛠 Installation
1. Clone the repository
git clone https://github.com/shermack/edge-signtalk-real-time-sign-language-interpreter.git
cd edge-signtalk-real-time-sign-language-interpreter/tflite-model

2. Install dependencies
pip install -r requirements.txt

3. Run the full GUI
python silent_voice.py

4. Optional: run minimal TFLite test
python run_interpreter.py

🧩 How It Works
1. Capture frame → detect hand using MediaPipe

MediaPipe Hands extracts the bounding box of the detected hand.

2. Crop + resize to 96×96 grayscale
crop = crop_hand(gray, x, y, w, h)
crop = cv2.resize(crop, (96, 96))
crop = crop.reshape(1, 96, 96, 1)
crop_int8 = (crop / 2 - 128).astype(np.int8)  # quantization scaling

3. Send cropped frame into INT8 TFLite model

Classifier outputs softmax predictions over:

A–Z + SPACE + DELETE

4. Smooth predictions

A moving average filter ensures:

No random jumping

Only stable poses result in a predicted letter

5. UI displays predicted letter + overlays landmarks

Users see both:

The classification

The skeletal hand visualization

📊 Model Summary
Property	Value
Model type	Image Classification (Transfer Learning)
Input	96 × 96 (grayscale)
Output	28 ASL letters
Quantization	INT8 (fully quantized)
Test accuracy	95.86%
Dataset	Custom ASL dataset + Edge Impulse test tool
🧪 Known Limitations & Future Work
🔹 Single-hand only

Two-hand support requires a different dataset.

🔹 Lighting conditions

Extreme darkness can reduce accuracy.

🔹 Sentence Builder Mode

Planned but excluded from final hackathon build due to time.

🚀 Future Enhancements

Two-hand signs (e.g., “H”, “K”, “R”)

Dynamic signs (motion-based like “J”)

Voice synthesis of predicted sentences

Mobile Android/iOS deployment

Edge TPU version (Coral USB / DevBoard)

📝 License

MIT License — free for commercial and research use.

👤 Author

Shermack Salat
Nairobi, Kenya

🤝 Acknowledgements

Edge Impulse — training pipeline & model optimization

MediaPipe — hand tracking

TensorFlow Lite — on-device inference
