# AUV Vision – Image Preprocessing Pipeline
This module prepares underwater images of gates for robust object detection by applying contrast enhancement and sharpening techniques.
> Built for the Autonomous Underwater Vehicle (AUV) Machine Learning task

---

## Overview

Underwater images often suffer from low visibility, noise, and poor edge definition. Traditional denoising methods were found to blur gate boundaries. This pipeline enhances image clarity by using:

- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** for local contrast enhancement  
- **Sharpening filters** to amplify structural features like gate edges

These enhanced images are used as training data for the YOLOv8 object detection model.

---

## Sample Results
RAW IMAGE:	<img width="195" height="195" alt="image" src="https://github.com/user-attachments/assets/e93d07b3-859c-4062-b129-2c67b1ee047f" />

AFTER DENOISING: <img width="190" height="190" alt="image" src="https://github.com/user-attachments/assets/7e09a5a2-747e-4bd6-80d2-4a5c16845b1a" />

AFTER CONTRASTING:<img width="204" height="204" alt="image" src="https://github.com/user-attachments/assets/0fe0898e-0c10-437f-a21e-1bd18e57f74f" />

AFTER SHARPENING:<img width="204" height="204" alt="image" src="https://github.com/user-attachments/assets/94aea8f3-2b76-4bfd-959a-5eeff46bc4e9" />

---

## How to Run
### 1. Setup

Install dependencies in a virtual environment:
python3 -m venv venv
source venv/bin/activate
pip install opencv-python numpy

### 2. 📜 Run CLAHE only:
python scripts/clahe.py

### 3. 📜 Run CLAHE + Sharpening:
python scripts/sharpen.py

---
**Why This Matters**
- Denoising blurred edge features and reduced YOLO accuracy
- CLAHE improves local contrast in murky water
- Sharpening brings out gate boundaries
- Greatly improves input quality for training object detection models like YOLOv8

**Next Step**
Use the sharpened_dataset/ as the input to your object detection pipeline (e.g., Roboflow → YOLOv8 training).

**CONCLUSION:**
After experimenting with multiple preprocessing techniques, CLAHE-based contrast enhancement combined with light sharpening provided the most effective results for underwater gate images. Unlike traditional denoising, which blurred critical features, this approach improved edge clarity without loss of detail — significantly aiding visual recognition. These enhanced images form the foundation for robust YOLOv8-based object detection and anomaly filtering in future training phases.
