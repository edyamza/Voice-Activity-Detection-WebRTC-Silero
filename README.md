# 🗣️ Voice Activity Detection with WebRTC & Silero

A cross-model Voice Activity Detection (VAD) tool with real-time and file-based analysis using both **WebRTC VAD** and **Silero VAD** models, built in **PyQt6**. Designed for visualization, evaluation, and comparison.

---

## 🚀 Features

- 🎙️ **Live microphone-based VAD**
- 📂 **File-based VAD analysis**
- 📊 **Comparison mode**: WebRTC vs Silero metrics side by side
- 📈 Generates:
  - Spectrograms
  - Waveforms
  - Confusion matrices
  - Metric charts (Accuracy, Precision, Recall, F1-score, etc.)
- 🧠 Silero model with intelligent frame-by-frame classification
- 🌐 WebRTC model integrated for fast binary VAD

---

## 🧰 Tech Stack

- Python 3.9+
- PyQt6
- matplotlib, numpy
- simpleaudio
- wave
- WebRTC VAD wrapper
- Silero VAD

---

## 💻 Installation

1. **Clone the repo**

```bash
git clone https://github.com/edyamza/Voice-Activity-Detection-WebRTC-Silero.git
cd Voice-Activity-Detection-WebRTC-Silero
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
python vad_guide.py
```

---

## 🖼️ GUI Preview

> 🟢 Live status color changes  
> 📈 Spectrograms for both models  
> 📊 Comparison window auto-generates performance bar charts



---

## 📁 Project Structure

```
├── vad_guide.py            # Main GUI application
├── vad_rec.py              # WebRTC file-based VAD
├── vad_rec_silero.py       # Silero VAD interface
├── vad_live.py             # Live audio processing
├── evaluation.py           # Metrics & graph generation
├── output/                 # Saved plots and images
└── requirements.txt
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## ✨ Author

**Eduard Amza** — [GitHub](https://github.com/edyamza)

---

## 🧠 Inspired by

- [Silero VAD](https://github.com/snakers4/silero-vad)
- [WebRTC VAD](https://github.com/wiseman/py-webrtcvad)

---

Feel free to ⭐ the project or contribute!
