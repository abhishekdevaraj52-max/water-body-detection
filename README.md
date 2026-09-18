# 🌊 Water Body Segmentation AI

AI-powered water body detection using a **U-Net deep learning model** trained on **2841 satellite images**.

---

## 📁 Project Structure

```
project/
│
├── server.py               ← Flask backend (API + UI server)
├── model.py                ← U-Net model architecture
├── predict.py              ← Standalone prediction script
├── app.py                  ← Alternative app entry point
│
├── templates/
│   └── index.html          ← Frontend UI (drag & drop interface)
│
├── models/
│   └── water_unet.h5       ← Trained model (2841 images)
│
├── train_all.py            ← Train on ALL images (recommended)
├── train_cpu_1000.py       ← Train first 1000 images
├── train_cpu_next_1000.py  ← Train next 1000 images
├── train_fast.py           ← Quick test (40 images)
│
├── requirements.txt        ← Python dependencies
├── START_SERVER.bat        ← ⭐ One-click launcher (Windows)
└── README.md               ← This file
```

---

## 🚀 Quick Start (Windows)

### Option 1 — Double-click to start ⭐
Just double-click `START_SERVER.bat` — it will:
- Check Python is installed
- Install requirements if missing
- Start the server
- Open the browser automatically

### Option 2 — Manual command
```bash
cd "Water Bodise images\project"
python server.py
```
Then open: **http://127.0.0.1:5050**

---

## 🖥️ How to Use the UI

1. Open **http://127.0.0.1:5050** in your browser
2. **Drag & drop** a satellite or aerial image (JPG/PNG)
3. Click **"Detect Water Bodies"**
4. View the **segmentation overlay** and **water coverage %**
5. Click **Download** to save the result

---

## 🧠 Training the Model

```bash
# Train on ALL 2841 images (continues from existing model)
python train_all.py

# Train only the first 1000 (fresh start)
python train_cpu_1000.py

# Train next 1000 (requires first run to be done)
python train_cpu_next_1000.py

# Quick 40-image test
python train_fast.py
```

---

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🌐 Deploy Online (Free Options)

| Platform | Type | Free Tier |
|---|---|---|
| [Render](https://render.com) | Backend + Frontend | ✅ Yes |
| [Railway](https://railway.app) | Backend | ✅ Yes |
| [PythonAnywhere](https://pythonanywhere.com) | Flask apps | ✅ Yes |
| [Hugging Face Spaces](https://huggingface.co/spaces) | ML apps | ✅ Yes |

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Architecture | U-Net |
| Input Size | 512×512 |
| Training Images | 2841 |
| Final Val Accuracy | ~69% |
| Final Val Loss | 0.627 |

---

## 🔧 Tech Stack

- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Backend**: Python, Flask
- **Model**: TensorFlow/Keras U-Net
- **Image Processing**: OpenCV, NumPy
