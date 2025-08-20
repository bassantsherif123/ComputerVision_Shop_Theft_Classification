import torch
from .model_def import get_model
from ultralytics import YOLO
import cv2
import os
from django.conf import settings
import subprocess
import imageio_ffmpeg

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv8 (person detection)
yolo_model = YOLO("yolov8n.pt")

CLASSES = ["Non Shoplifter", "Shoplifter"]

def load_model(model_name):
    model_path = os.path.join(os.path.dirname(__file__), "models", model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = get_model(model_name)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def run_inference(video_path, model_name="best_model.pth"):
    # --- 1. Load classifier dynamically ---
    model = load_model(model_name)

    model.eval()    
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Default preprocessing (match training)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        frame_tensor = torch.tensor(frame_resized).permute(2, 0, 1).float() / 255.0
        frames.append(frame_tensor)
    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames found in video.")

    # Stack into shape (1, T, C, H, W)
    video_tensor = torch.stack(frames).unsqueeze(0).float()
    
    threshold = 0.5
    with torch.no_grad():
        if model_name.startswith("efficientnet_lstm"):
            outputs = model(video_tensor).squeeze(1)
            confidence = torch.sigmoid(outputs).item()
            prediction = "Shoplifter" if confidence > threshold else "Non Shoplifter"
        elif model_name.startswith("custom_3dcnn"):
            # Rearrange → (1, C, T, H, W) for 3D CNN
            video_tensor = video_tensor.permute(0, 2, 1, 3, 4).to(device)
            logits = model(video_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            prediction = CLASSES[pred_idx]
            confidence = probs.max().item()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    

    # --- 2. YOLOv8 Detection ---
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(settings.MEDIA_ROOT, "output_detected_raw.mp4")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        # Run YOLO person detection (no labels)
        results = yolo_model.predict(frame, classes=[0], verbose=False)  # class 0 = person
    
        # Draw green boxes without labels
        for box in results[0].boxes.xyxy:  # xyxy format
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)  # Green box
    
        out.write(frame)
    
    cap.release()
    out.release()
    
    final_path = os.path.join(settings.MEDIA_ROOT, "output_detected.mp4")
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()  # ensures venv ffmpeg is used
    command = [
        ffmpeg_bin, "-y", "-i", out_path,
        "-vcodec", "libx264", "-acodec", "aac",
        "-strict", "experimental", final_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return prediction, confidence, final_path
