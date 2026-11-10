import os
import torch
import torch.nn as nn
import torchvision.models as models


class DeepfakeDetector(nn.Module):
    """
    TruthVision Deepfake Detection Neural Network.
    Supports transfer learning backbones (EfficientNet-B0, ResNet50, MobileNetV3)
    with custom classification head and Grad-CAM target layer access.
    """
    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True, num_classes: int = 1, dropout_rate: float = 0.4):
        super(DeepfakeDetector, self).__init__()
        self.backbone_name = backbone.lower()
        self.num_classes = num_classes

        if self.backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b0(weights=weights)
            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
            self.feature_extractor = base_model
            # For Grad-CAM target layer
            self.target_layer = self.feature_extractor.features[-1]

        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.feature_extractor = base_model
            self.target_layer = self.feature_extractor.layer4[-1]

        elif self.backbone_name == "mobilenet_v3":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base_model = models.mobilenet_v3_small(weights=weights)
            in_features = base_model.classifier[0].in_features
            base_model.classifier = nn.Identity()
            self.feature_extractor = base_model
            self.target_layer = self.feature_extractor.features[-1]

        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from 'efficientnet_b0', 'resnet50', 'mobilenet_v3'.")

        # Custom high-capacity classification head with dropout & regularization
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns raw logits (for BCEWithLogitsLoss).
        """
        features = self.feature_extractor(x)
        # In case features are 4D (batch, channels, H, W)
        if len(features.shape) > 2:
            features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates sigmoid probability for deepfake class (0.0 = Real, 1.0 = Fake).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if self.num_classes == 1:
                return torch.sigmoid(logits)
            return torch.softmax(logits, dim=1)

    def get_target_layer(self):
        """Returns target convolutional layer for Grad-CAM explainability."""
        return self.target_layer


def load_model(checkpoint_path: str = None, backbone: str = "efficientnet_b0", device: str = "cpu") -> DeepfakeDetector:
    """
    Factory helper to instantiate and load weights onto target device.
    """
    model = DeepfakeDetector(backbone=backbone, pretrained=(checkpoint_path is None or not os.path.exists(checkpoint_path)))
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Handle state dict wrapped in 'model_state_dict'
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model
