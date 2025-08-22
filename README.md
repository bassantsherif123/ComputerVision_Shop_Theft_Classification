# 🛒 Shop Theft Detection System  

## 📌 Objective  
This project aims to build an AI system that predicts whether **shoplifting has occurred in a surveillance video**.  
It combines **deep learning video classification models** with **YOLOv8 person detection** to create an end-to-end solution for automated shop theft monitoring.  


## 🚀 Features  
- **Multiple Video Classifiers**:  
  Includes different deep learning approaches.  

- **YOLOv8 Person Detection**:  
  Detects people frame-by-frame in videos and highlights them with bounding boxes.  

- **Django Web Deployment**:  
  Upload a video → system predicts whether it contains shoplifting → outputs a video with people detections.  


## 🧠 Models  
The system includes multiple deep learning models for video classification:  

- **EfficientNetB0 + LSTM** – Combines spatial and temporal features.  
- **Simple 3D CNN** – Learns spatio-temporal patterns directly from video.  
- **Video Transformer (ResNet + Transformer)** – Extracts frame features with ResNet and models sequences using a Transformer.  
- **VideoMAE (HuggingFace)** – Pretrained transformer for video understanding.

## ⚙️ Deployment Workflow
- User uploads a video in the Django web app.
- User Choose a Model from `EfficientNetB0` + LSTM and `Custom 3D CNN`.
- Frames are extracted and passed into the chosen video classifier.
- YOLOv8 detects people in frames and draws bounding boxes.
- Output:
    - Prediction (Shoplifter / Non-Shoplifter with confidence).
    - A Video showing detected people.

## 📂 Project Structure
```
ComputerVision_Shop_Theft_Classification/
|── Deployment/ProjectUI/
│   │── pages/ 
|   |   │── models/                   
│   |   ├── views.py
│   |   ├── urls.py
│   |   ├── tests.py
│   |   ├── apps.py
│   |   ├── models.py            
│   |   ├── inference.py        
│   |   ├── model_def.py                  
|   |   ProjectUI/
│   |   ├── static/
│   |   ├── __init.py__
│   |   ├── asgi.py
│   |   ├── settings.py
│   |   ├── urls.py
│   |   ├── wsgi.py         
|   |── templates/
|   │── requirements.txt        # Dependencies  
│── notebooks/              
│── README.md               # Project description
```

## 🛠️ Installation
### Clone repo
```
git clone https://github.com/bassantsherif123/ComputerVision_Shop_Theft_Classification.git
cd shop-theft-detection
```

### Create environment
```
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv/Scripts/activate     # (Windows)
```
### Install dependencies
```
pip install -r requirements.txt
```

## ▶️ Run Django App
```
cd Deployment/ProjectUI
python manage.py collectstatic
python manage.py runserver
```
Open in browser: http://127.0.0.1:8000/
___
**_This project was done in collaboration with my colleague [Duaa Swalmeh](https://github.com/Duaa-Swalmeh)_**