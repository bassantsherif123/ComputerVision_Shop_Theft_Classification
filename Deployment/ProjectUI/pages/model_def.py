import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# EfficientNet + LSTM
# -----------------------------

class EfficientNetB0_LSTM(nn.Module):
    def __init__(self, hidden_size=128, num_classes=1):
        super().__init__()
        
        # Pretrained EfficientNetB0 backbone
        backbone = efficientnet_b0(pretrained=True)
        backbone.classifier = nn.Identity()  # remove classification head
        self.backbone = backbone
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Sequence modeling
        self.lstm = nn.LSTM(
            input_size=1280,  # EfficientNetB0 output feature size
            hidden_size=hidden_size,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch, time, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)              # merge batch & time for CNN
        
        feats = self.backbone(x)                # (b*t, 1280)
        feats = feats.view(b, t, -1)             # (batch, time, 1280)
        
        _, (h_n, _) = self.lstm(feats)           # h_n: (1, batch, hidden_size)
        h_n = h_n.squeeze(0)
        
        out = self.dropout(h_n)
        out = self.fc(out)
        
        # Binary Classification
        return torch.sigmoid(out)
    
# -----------------------------
# Simple 3D CNN
# -----------------------------
class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1,1,1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):  # x: (B, C, T, H, W)
        x = self.features(x)
        x = self.classifier(x)
        return x
        
# -----------------------------
# Helper: model factory
# -----------------------------
def get_model(name: str):
    if name.startswith('efficientnet_lstm'):
        return EfficientNetB0_LSTM().to(device)
    elif name.startswith('custom_3dcnn'):
        return Simple3DCNN().to(device)
    else:
        raise ValueError(f'Unknown model type: {name}')
