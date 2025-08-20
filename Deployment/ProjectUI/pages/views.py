from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
import os
from .inference import run_inference
from django.conf import settings

MODEL_CHOICES = {
    "efficientnet_lstm": "efficientnet_lstm.pth",
    "custom_3dcnn": "custom_3dcnn.pth",
}

def index(request):
    return render(request, "pages/index.html")

def predict(request):
    if request.method == "POST":
        video_file = request.FILES["video"]
        
        model_name = request.POST.get("model")
        print("Selected model:", model_name)
        
        # Save uploaded file
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(video_file.name, video_file)
        video_path = fs.path(filename)

        # Run inference
        prediction, confidence, output_video_path = run_inference(video_path, model_name)

        return JsonResponse({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "video_url": fs.url(os.path.basename(output_video_path)),
        })
    return JsonResponse({"error": "No video uploaded"}, status=400)
