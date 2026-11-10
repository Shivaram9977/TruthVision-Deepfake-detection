import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    Visualizes high-activation feature regions contributing to deepfake classification.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate(self, input_tensor: torch.Tensor, class_idx: int = 0) -> np.ndarray:
        """
        Computes Grad-CAM 2D heatmap [0.0 - 1.0] for the input tensor.
        """
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)
        if logits.shape[-1] == 1:
            score = logits[0, 0]
        else:
            score = logits[0, class_idx]

        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # Global average pooling on gradients across spatial dimensions
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H, W]

        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Handle edge cases where all activations are zero
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)


def generate_heatmap_overlay(
    cam: np.ndarray,
    original_rgb: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Blends Grad-CAM heatmap over original RGB image.
    Returns: (blended_rgb, raw_heatmap_rgb)
    """
    h, w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Convert to 8-bit heatmap
    heatmap_uint8 = np.uint8(255 * cam_resized)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    blended = np.uint8(alpha * heatmap_rgb + (1 - alpha) * original_rgb)
    return blended, heatmap_rgb


def analyze_activation_regions(cam: np.ndarray) -> Dict[str, Any]:
    """
    Analyzes spatial distribution of activation peaks to identify likely manipulation zones.
    Divided into:
    - Upper face (Forehead / Eyes / Brows)
    - Middle face (Nose / Cheeks)
    - Lower face (Mouth / Lips / Chin)
    - Perimeter / Boundaries (Blending seam artifacts)
    """
    h, w = cam.shape
    threshold = 0.6
    high_act_mask = cam >= threshold

    total_high_act = np.sum(high_act_mask)
    if total_high_act == 0:
        return {
            "suspicious_zones": ["Uniform / Low Artifact"],
            "boundary_artifact_ratio": 0.0,
            "focal_region": "None",
            "artifact_severity": "Low"
        }

    # Region slices
    upper_mask = high_act_mask[: int(h * 0.35), :]
    mid_mask = high_act_mask[int(h * 0.35): int(h * 0.65), :]
    lower_mask = high_act_mask[int(h * 0.65):, :]

    # Perimeter mask (outer 15% border)
    border_h, border_w = int(h * 0.15), int(w * 0.15)
    perimeter_mask = np.zeros_like(high_act_mask, dtype=bool)
    perimeter_mask[:border_h, :] = True
    perimeter_mask[-border_h:, :] = True
    perimeter_mask[:, :border_w] = True
    perimeter_mask[:, -border_w:] = True
    boundary_acts = np.sum(high_act_mask & perimeter_mask)

    upper_count = np.sum(upper_mask)
    mid_count = np.sum(mid_mask)
    lower_count = np.sum(lower_mask)

    suspicious_zones: List[str] = []
    if upper_count / total_high_act > 0.35:
        suspicious_zones.append("Eyes / Orbital Region & Forehead")
    if mid_count / total_high_act > 0.35:
        suspicious_zones.append("Nose & Facial Texture Surface")
    if lower_count / total_high_act > 0.35:
        suspicious_zones.append("Mouth / Lips & Mandibular Boundary")

    boundary_ratio = float(boundary_acts / (total_high_act + 1e-6))
    if boundary_ratio > 0.30:
        suspicious_zones.append("Facial Blending Boundary Seam")

    if not suspicious_zones:
        suspicious_zones.append("Diffuse Artifacts Across Face")

    counts = {"Upper": upper_count, "Middle": mid_count, "Lower": lower_count}
    focal_region = max(counts, key=counts.get)

    if total_high_act / (h * w) > 0.25 or boundary_ratio > 0.40:
        severity = "High"
    elif total_high_act / (h * w) > 0.10:
        severity = "Moderate"
    else:
        severity = "Low"

    return {
        "suspicious_zones": suspicious_zones,
        "boundary_artifact_ratio": round(boundary_ratio * 100, 1),
        "focal_region": focal_region,
        "artifact_severity": severity,
        "saliency_density": round(float(total_high_act / (h * w)) * 100, 1)
    }
