import os
import cv2
import numpy as np
import torch
from typing import Dict, Any, Optional
from PIL import Image

from models.detector import load_model, DeepfakeDetector
from preprocessing.face_detector import FaceDetector, preprocess_image_or_crop
from explainability.gradcam import GradCAM, generate_heatmap_overlay, analyze_activation_regions
from forensics.risk_scorer import ForensicRiskScorer, ForensicAssessment


def predict_image(
    image_path: str,
    model: Optional[DeepfakeDetector] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Full pipeline for single image deepfake prediction:
    1. Detect and crop face
    2. Model forward pass -> Fake Probability
    3. Grad-CAM saliency extraction & overlay
    4. Saliency region identification
    5. Multi-factor forensic scoring
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None:
        model = load_model(checkpoint_path=checkpoint_path, device=device)
    else:
        model.to(device)

    detector = FaceDetector()
    tensor, face_crop_rgb, bbox = preprocess_image_or_crop(image_path, img_size=224, detector=detector)
    tensor = tensor.to(device)

    # Prediction
    with torch.no_grad():
        proba = float(model.predict_proba(tensor).item())

    # Grad-CAM explainability
    gradcam_engine = GradCAM(model, model.get_target_layer())
    cam = gradcam_engine.generate(tensor)
    gradcam_engine.remove_hooks()

    blended_overlay, raw_heatmap = generate_heatmap_overlay(cam, face_crop_rgb, alpha=0.5)
    saliency_analysis = analyze_activation_regions(cam)

    # Forensic Assessment
    scorer = ForensicRiskScorer()
    media_name = os.path.basename(image_path)
    assessment: ForensicAssessment = scorer.evaluate_image(
        media_name=media_name,
        deepfake_prob=proba,
        rgb_crop=face_crop_rgb,
        saliency_data=saliency_analysis
    )

    # Save visualization assets if output_dir provided
    saved_paths = {}
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(media_name)[0]
        crop_path = os.path.join(output_dir, f"{base}_crop.jpg")
        overlay_path = os.path.join(output_dir, f"{base}_gradcam.jpg")
        heatmap_path = os.path.join(output_dir, f"{base}_heatmap.jpg")

        cv2.imwrite(crop_path, cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(overlay_path, cv2.cvtColor(blended_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(heatmap_path, cv2.cvtColor(raw_heatmap, cv2.COLOR_RGB2BGR))

        saved_paths = {
            "face_crop": crop_path,
            "gradcam_overlay": overlay_path,
            "heatmap": heatmap_path
        }

    return {
        "media_name": media_name,
        "classification": assessment.classification,
        "confidence": round(assessment.deepfake_probability if assessment.classification == "DEEPFAKE" else assessment.authenticity_score, 2),
        "deepfake_probability": round(proba * 100.0, 2),
        "authenticity_score": round(assessment.authenticity_score, 2),
        "risk_level": assessment.risk_level,
        "risk_score": assessment.risk_score,
        "likely_technique": assessment.likely_technique,
        "suspicious_zones": assessment.suspicious_regions,
        "saliency_analysis": saliency_analysis,
        "bbox": bbox,
        "face_crop_rgb": face_crop_rgb,
        "gradcam_overlay_rgb": blended_overlay,
        "raw_heatmap_rgb": raw_heatmap,
        "saved_paths": saved_paths,
        "assessment": assessment.to_dict()
    }
