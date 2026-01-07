# Scientific-and-implementation-project-2

# Personal Colour Analysis System

This is an application that uses **Machine Learning (SVM)** to perform **Personal Colour Analysis (PCoA)**. Users can predict their seasonal color type (Spring, Summer, Autumn, Winter) based on an image from their folder or directly from the camera.

## 🔍 Project Overview

The application takes an input image, applies preprocessing including face detection and color extraction, and then uses a trained SVM model to classify the user into a seasonal color type. The system is designed as a ready-to-use tool for personal colour analysis.

## Key Features

* Accepts images from local files or live camera input
* Preprocessing of facial images including **white balancing**
* Uses **SVM classifier** for color type prediction
* Fast and user-friendly application

# How to run the app 

## macOS / Linux

### Clone the repository

```bash
git clone https://github.com/Gosiuniunia/Scientific-and-implementation-project-2.git
cd Scientific-and-implementation-project-2
```

### Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Download the model

Download the face landmark model from [Mediapipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker?hl=pl) and place it in the `models` folder.

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Run the application

```bash
python ./run_pcoa_app.py
```

---

## Windows

### Clone the repository

```bash
git clone https://github.com/Gosiuniunia/Scientific-and-implementation-project-2.git
cd Scientific-and-implementation-project-2
```

### Create and activate a virtual environment (PowerShell)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Download the model

Download the face landmark model from [Mediapipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker?hl=pl) and place it in the `models` folder.

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Run the application

```bash
python ./run_pcoa_app.py
```


No matter the OS, paste into your browser of choice url shown in console, for example:
* Running on local URL:  http://127.0.0.1:7860
