# Scientific-and-implementation-project-2

# Personal Colour Analysis System

ins implementations of machine learning (ML) and deep learning (DL) methods for **Personal Colour Analysis (PCoA)** system.  
The goal of this project is to classify individuals into seasonal color types (Spring, Summer, Autumn, Winter) based on image data.

## 🔍 Project Overview TODO: update

The environment incorporates various components including color feature extraction (based on facial images and landmark detection), hyperparameter tuning of ML classifiers (KNN, SVM, Decision Trees), and preparation and training of DL models (e.g., VGG16), with support for data augmentation.

Additionally, the environment provides tools for testing and comparing classifier performance using statistical tests.

## Key Features
- Image preprocessing including **White balancing**
- Two distinct modelling approaches:

  1. **Feature-based Machine Learning**:
     - Extraction of dominant colours from key facial regions (eyes, skin, eyebrows) using MediaPipe facial landmarks
     - Application of classical ML algorithms:
       - Support Vector Machine (SVM)

# How to run the app - macOS/Linux

```bash
git clone https://github.com/Gosiuniunia/Scientific-and-implementation-project-2.git
cd pcoa_app
source .venv/bin/activate
python3 -m pip install app_requirements.txt
python3 run_pcoa_app.py
```

# How to run the app - Windows
```bash
git clone https://github.com/Gosiuniunia/Scientific-and-implementation-project-2.git
cd pcoa_app
.venv/Scripts/Activate.ps1 # for PowerShell users
python -m pip install app_requirements.txt
python run_pcoa_app.py
```

No matter the OS, paste into your browser of choice url shown in console, for example:
* Running on local URL:  http://127.0.0.1:7860