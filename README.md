# 🩸 Advanced Fingerprint-Based Blood Group Classification System

A state-of-the-art Deep Learning project designed to identify human blood groups using unique fingerprint patterns. This project integrates **Computer Vision**, **CNN Architectures**, and a **Streamlit Web Interface** to provide real-time biometric analysis.

---

## 📌 Project Overview
Biometric patterns like fingerprints are traditionally used for identification, but recent studies suggest correlations with certain physiological traits. This project implements a **Convolutional Neural Network (CNN)** to classify fingerprints into 8 distinct blood groups: **A+, A-, B+, B-, AB+, AB-, O+, and O-**.

## 🧠 Technical Architecture

### 1. Data Pipeline & Preprocessing
* **Format Handling:** The system specifically handles `.BMP` (Bitmap) files, which are uncompressed and provide the high ridge-definition necessary for fingerprint analysis.
* **Color Space Conversion:** To prevent "Channel Mismatch" errors (RGBA/4-channel issues), all input images are dynamically converted to **RGB (3-channel)**.
* **Normalization:** Pixel values are scaled from [0, 255] to **[0.0, 1.0]** to ensure faster convergence during gradient descent.
* **Image Resizing:** Standardized input size of **180x180 pixels** to maintain a balance between computational speed and feature preservation.

### 2. Deep Learning Model (CNN)
The model is built using **TensorFlow/Keras** with the following layers:
* **Convolutional Layers:** Multiple `Conv2D` layers with 3x3 filters to extract spatial features like minutiae, ridge endings, and bifurcations.
* **Regularization:** Integrated **Dropout (0.5)** to combat overfitting, ensuring the model generalizes well on unseen data.
* **Data Augmentation:** Real-time transformations including `RandomRotation` and `RandomZoom` are applied during training to make the model invariant to finger orientation.
* **Optimization:** **Adam Optimizer** with a categorical cross-entropy loss function.

### 3. Frontend & Deployment
* **Streamlit Dashboard:** A responsive web application that allows users to upload fingerprint images and get instant predictions.
* **Performance Metrics:** Displays a **Confidence Progress Bar** to indicate the model's certainty.

---

## 🛠️ Installation & Environment Setup

### Prerequisites
* Python 3.9+
* Anaconda (Recommended)

### Steps to Install
1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/Amanpandit87/fingerprint-blood-group-detector.git](https://github.com/Amanpandit87/fingerprint-blood-group-detector.git)
   cd fingerprint-blood-group-detector

### 🛠️ Installation & Execution

**1. Install Compatible Dependencies:**
> **Note:** Using `numpy < 2.0` is critical for TensorFlow 2.15 compatibility to avoid "umath" import errors.

```bash
pip install tensorflow==2.15.0 "numpy<2" opencv-python pillow streamlit matplotlib

# Blood-group-prediction
An AI-powered Biometric Classification system that predicts human blood groups from fingerprint patterns using Deep Learning. Built with a custom CNN architecture in TensorFlow and a real-time Streamlit dashboard, this project achieves 87% validation accuracy by analyzing unique ridge minutiae.
