# 🎵 AI Harmonica
### by **Eliiviz**

> Identify every musical instrument in any song using a CNN ensemble of CNN14 + CNN (trained on OpenMIC).

![AI Harmonica](https://img.shields.io/badge/AI%20Harmonica-Eliiviz-8b5cf6?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi)
![React](https://img.shields.io/badge/React-18-61dafb?style=for-the-badge&logo=react)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.x-06b6d4?style=for-the-badge&logo=tailwindcss)

---

## 🚀 Features

- 🎸 **Dual CNN Ensemble** — CNN14 (AudioSet, 527 classes) + CNN (trained on OpenMIC, 20 classes) combined via weighted score fusion
- ⭐ **"Confirmed by Both Models" section** — instruments detected by both networks are highlighted at the top
- 🎛️ **Adjustable ensemble weights** — slider to balance CNN14 vs CNN influence in real-time
- 📊 **Instrument Timeline** — temporal breakdown showing when each instrument appears
- 🎵 **Built-in audio player** — with waveform visualisation and seek
- 🌑 **Stunning dark UI** — deep navy/indigo glassmorphism theme

---

## 🏗️ Project Structure

```
AI Harmonica/
├── backend/                  # FastAPI Python backend
│   ├── main.py               # API endpoints
│   ├── ensemble_classifier.py# CNN14 + CNN ensemble inference
│   ├── classifier.py         # CNN14 wrapper
│   ├── instruments_data.py   # Instrument metadata catalog
│   ├── requirements.txt
│   └── My CNN backend/       # Custom CNN trained on OpenMIC
│       ├── models/
│       │   └── openmic_instrument_cnn.pth  # trained weights
│       ├── train_cnn.py
│       └── ...
└── frontend/                 # Vite + React + Tailwind CSS
    ├── src/
    │   ├── App.jsx
    │   ├── components/
    │   │   ├── Navbar.jsx
    │   │   ├── AudioDropzone.jsx
    │   │   ├── AudioPlayerWaveform.jsx
    │   │   ├── InstrumentCard.jsx
    │   │   ├── SongMetrics.jsx
    │   │   ├── TimelineVisualizer.jsx
    │   │   └── EnsembleWeightPanel.jsx
    │   └── index.css
    ├── index.html
    └── package.json
```

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
```

**Download PANNs CNN14 checkpoint** (not included due to size):
```bash
# Download CNN14_mAP=0.431.pth from https://zenodo.org/record/3987831
# Place it wherever your classifier.py expects it
```

**Start the API:**
```bash
# From the project root
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**

---

## 🎛️ API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check, device info |
| `GET` | `/api/demos` | List demo tracks |
| `GET` | `/api/demos/{id}` | Stream demo audio file |
| `POST` | `/api/analyze?cnn14_weight=0.6&openmic_weight=0.4` | Analyze uploaded audio |
| `POST` | `/api/analyze-demo/{id}?cnn14_weight=0.6&openmic_weight=0.4` | Analyze demo track |

---

## 🧠 Models

| Model | Dataset | Classes | Purpose |
|-------|---------|---------|---------|
| **CNN14** | Google AudioSet | 527 | Broad instrument + sound coverage |
| **CNN (trained on OpenMIC)** | OpenMIC-2018 | 20 | Precise musical instrument recall |

Results are fused using **weighted score fusion** — each instrument gets a combined score based on both model outputs.

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vite, React 18, Tailwind CSS |
| Backend | FastAPI, Python 3.10+ |
| ML Models | PyTorch, torchaudio, librosa |
| Audio | PANNs CNN14, Custom OpenMIC CNN |

---

## 📄 License

MIT License © 2026 Eliiviz
