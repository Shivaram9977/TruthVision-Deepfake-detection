import os
import cv2
import numpy as np
import torch
from typing import Dict, Any, List, Optional
from PIL import Image

from models.detector import load_model, DeepfakeDetector
from preprocessing.face_detector import FaceDetector, get_default_transform
from explainability.gradcam import GradCAM, generate_heatmap_overlay, analyze_activation_regions
from forensics.risk_scorer import ForensicRiskScorer, ForensicAssessment


class VideoInferenceEngine:
    """
    Asynchronous and batch video analysis engine.
    Extracts representative frames, processes faces, traces prediction timeline,
    and pinpoints anomaly video intervals.
    """
    def __init__(self, model: Optional[DeepfakeDetector] = None, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if model is None:
            self.model = load_model(checkpoint_path=checkpoint_path, device=self.device)
        else:
            self.model = model.to(self.device)

        self.detector = FaceDetector()
        self.transform = get_default_transform(img_size=224)
        self.scorer = ForensicRiskScorer()

    def analyze_video(
        self,
        video_path: str,
        sample_rate_fps: float = 2.0,
        max_frames: int = 60,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs comprehensive frame-by-frame deepfake detection on video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video stream from {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_video_frames / fps

        # Determine frame stride based on desired sample rate
        frame_interval = max(1, int(fps / sample_rate_fps))

        frame_idx = 0
        extracted_count = 0
        timestamps: List[float] = []
        frame_probs: List[float] = []
        sampled_frames_data: List[Dict[str, Any]] = []

        peak_fake_prob = -1.0
        peak_frame_rgb = None
        peak_frame_tensor = None
        peak_timestamp = 0.0

        while cap.isOpened() and extracted_count < max_frames:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                current_time = frame_idx / fps
                timestamps.append(round(current_time, 2))

                # Face crop
                face_crop_rgb, bbox = self.detector.extract_face_crop(frame_bgr)
                pil_img = Image.fromarray(face_crop_rgb)
                tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    prob = float(self.model.predict_proba(tensor).item())

                frame_probs.append(prob)

                sampled_frames_data.append({
                    "frame_index": frame_idx,
                    "timestamp": round(current_time, 2),
                    "timestamp_str": f"{int(current_time // 60):02d}:{int(current_time % 60):02d}",
                    "deepfake_probability": round(prob * 100.0, 2),
                    "is_flagged": prob >= 0.50
                })

                if prob > peak_fake_prob:
                    peak_fake_prob = prob
                    peak_frame_rgb = face_crop_rgb
                    peak_frame_tensor = tensor
                    peak_timestamp = current_time

                extracted_count += 1

            frame_idx += 1

        cap.release()

        # Grad-CAM on peak suspicious frame
        peak_gradcam_overlay = None
        saliency_data = {}
        if peak_frame_tensor is not None and peak_frame_rgb is not None:
            gradcam_engine = GradCAM(self.model, self.model.get_target_layer())
            cam = gradcam_engine.generate(peak_frame_tensor)
            gradcam_engine.remove_hooks()
            peak_gradcam_overlay, _ = generate_heatmap_overlay(cam, peak_frame_rgb, alpha=0.5)
            saliency_data = analyze_activation_regions(cam)

        # Forensic Assessment
        media_name = os.path.basename(video_path)
        assessment = self.scorer.evaluate_video(
            media_name=media_name,
            frame_probabilities=frame_probs,
            timestamps=timestamps,
            avg_boundary_ratio=saliency_data.get("boundary_artifact_ratio", 0.0),
            avg_freq_anomaly=saliency_data.get("saliency_density", 0.0)
        )

        # Save artifacts if output_dir given
        saved_paths = {}
        if output_dir and peak_frame_rgb is not None:
            os.makedirs(output_dir, exist_ok=True)
            base = os.path.splitext(media_name)[0]
            peak_crop_path = os.path.join(output_dir, f"{base}_peak_crop.jpg")
            peak_overlay_path = os.path.join(output_dir, f"{base}_peak_gradcam.jpg")

            cv2.imwrite(peak_crop_path, cv2.cvtColor(peak_frame_rgb, cv2.COLOR_RGB2BGR))
            if peak_gradcam_overlay is not None:
                cv2.imwrite(peak_overlay_path, cv2.cvtColor(peak_gradcam_overlay, cv2.COLOR_RGB2BGR))

            saved_paths = {
                "peak_face_crop": peak_crop_path,
                "peak_gradcam_overlay": peak_overlay_path
            }

        return {
            "media_name": media_name,
            "classification": assessment.classification,
            "confidence": round(assessment.deepfake_probability if assessment.classification == "DEEPFAKE" else assessment.authenticity_score, 2),
            "deepfake_probability": round(assessment.deepfake_probability, 2),
            "authenticity_score": round(assessment.authenticity_score, 2),
            "risk_level": assessment.risk_level,
            "risk_score": assessment.risk_score,
            "likely_technique": assessment.likely_technique,
            "duration_sec": round(duration_sec, 2),
            "total_frames": total_video_frames,
            "frames_analyzed": len(frame_probs),
            "flagged_frames_count": assessment.flagged_frames_count,
            "flagged_intervals": assessment.flagged_intervals,
            "peak_fake_prob": round(peak_fake_prob * 100.0, 2),
            "peak_timestamp": round(peak_timestamp, 2),
            "timeline": sampled_frames_data,
            "timestamps": timestamps,
            "probabilities": [round(p * 100.0, 2) for p in frame_probs],
            "saliency_analysis": saliency_data,
            "saved_paths": saved_paths,
            "assessment": assessment.to_dict()
        }
