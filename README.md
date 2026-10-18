# 🛡️ DeepGuard AI – AI-Based Fake Video Detection System

## 📌 Introduction

DeepGuard AI is an AI-based application designed to identify potentially fake or manipulated videos using Machine Learning and Artificial Intelligence. It analyzes selected video frames using a pre-trained deepfake detection model and provides preliminary Real or Fake classification results.

The application uses Streamlit to provide a simple and interactive interface where users can upload videos, view prediction scores, and download a detection report.

## 🎯 Objectives

- Identify potentially manipulated video content using AI.
- Analyze sampled video frames using a pre-trained model.
- Display Real and Fake prediction scores.
- Provide a user-friendly interface for video analysis.
- Generate downloadable reports of detection results.

## ✨ Key Features

- 🎥 Video upload support for MP4, AVI, MOV, and MKV formats.
- 🤖 AI-powered frame classification.
- 🔍 Analysis of up to 8 sampled video frames.
- 📊 Real and Fake prediction scores.
- 🚦 Preliminary classification based on aggregated frame results.
- 📄 Downloadable text report.
- 🌐 Interactive Streamlit web interface.

## 🛠️ Technologies Used

- **Python** – Application development
- **Streamlit** – Web application interface
- **Hugging Face Transformers** – Pre-trained AI model integration
- **PyTorch** – Deep learning framework
- **OpenCV** – Video processing and frame extraction
- **Pandas** – Organizing and analyzing prediction results

## 🧠 AI Model

**Pre-trained model:** `dima806/deepfake_vs_real_image_detection`

The model performs image classification on selected video frames. The system uses the frame-level predictions to calculate aggregated scores and generate a preliminary video classification.

Model source: https://huggingface.co/dima806/deepfake_vs_real_image_detection

## ⚙️ How It Works

1. Upload a video through the application.
2. Preview the uploaded video.
3. Extract up to 8 sampled frames.
4. Analyze each frame using the pre-trained AI model.
5. Calculate aggregated Real and Fake prediction scores.
6. Display the preliminary detection decision.
7. Download the result report for future reference.

## 🚀 Installation and Setup

### Step 1: Open Anaconda Prompt

Navigate to the project directory:

```bash
cd /d C:\Users\pc
```

### Step 2: Activate the Environment

```bash
conda activate deepguard
```

### Step 3: Install Required Libraries

```bash
python -m pip install streamlit transformers torch opencv-python pandas pillow
```

### Step 4: Run the Application

```bash
python -m streamlit run AI_Based_Fake_Video_Detection_System.py
```

### Step 5: Open in Browser

Open the local URL shown in the terminal, usually:

http://localhost:8501

**Note:** Run these commands from the directory containing your Python application file. Adjust the folder path or filename if yours is different.

## 📊 Expected Output

The application provides:

- Uploaded
