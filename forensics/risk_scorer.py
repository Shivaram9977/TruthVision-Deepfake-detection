import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import datetime


@dataclass
class ForensicAssessment:
    media_name: str
    media_type: str  # "IMAGE" or "VIDEO"
    classification: str  # "REAL" or "DEEPFAKE"
    deepfake_probability: float  # 0.0 to 100.0
    authenticity_score: float  # 0.0 to 100.0 (100 - fake_prob weighted)
    risk_level: str  # "LOW", "MODERATE", "HIGH", "CRITICAL"
    risk_score: float  # 0.0 to 100.0
    likely_technique: str  # e.g. "Face Swap", "Facial Reenactment", "GAN/Diffusion Synthesis", "None Detected"
    suspicious_regions: List[str] = field(default_factory=list)
    boundary_artifact_ratio: float = 0.0
    frequency_anomaly_score: float = 0.0
    temporal_variance: float = 0.0
    frames_analyzed: int = 1
    flagged_frames_count: int = 0
    flagged_intervals: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    summary_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "media_name": self.media_name,
            "media_type": self.media_type,
            "classification": self.classification,
            "deepfake_probability": round(self.deepfake_probability, 2),
            "authenticity_score": round(self.authenticity_score, 2),
            "risk_level": self.risk_level,
            "risk_score": round(self.risk_score, 2),
            "likely_technique": self.likely_technique,
            "suspicious_regions": self.suspicious_regions,
            "boundary_artifact_ratio": self.boundary_artifact_ratio,
            "frequency_anomaly_score": round(self.frequency_anomaly_score, 2),
            "temporal_variance": round(self.temporal_variance, 4),
            "frames_analyzed": self.frames_analyzed,
            "flagged_frames_count": self.flagged_frames_count,
            "flagged_intervals": self.flagged_intervals,
            "timestamp": self.timestamp,
            "summary_notes": self.summary_notes,
        }


class ForensicRiskScorer:
    """
    Computes multi-factor forensic authenticity risk index.
    Combines neural confidence, frequency domain anomalies, and spatial saliency.
    """
    def __init__(self, fake_threshold: float = 0.50):
        self.fake_threshold = fake_threshold

    def compute_frequency_anomaly(self, rgb_image: np.ndarray) -> float:
        """
        Performs 2D Fast Fourier Transform (FFT) on grayscale face image to detect
        checkerboard artifacts and spectral anomalies characteristic of generative upsamplers.
        Returns score from 0.0 to 100.0.
        """
        try:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)

            # Measure high frequency vs low frequency energy distribution
            h, w = gray.shape
            cy, cx = h // 2, w // 2
            r = min(h, w) // 6

            # Center mask for low frequencies
            y, x = np.ogrid[:h, :w]
            mask = (x - cx)**2 + (y - cy)**2 <= r*r

            low_freq_energy = np.mean(magnitude_spectrum[mask])
            high_freq_energy = np.mean(magnitude_spectrum[~mask])

            ratio = float(high_freq_energy / (low_freq_energy + 1e-6))
            # Normalized anomaly indicator
            score = np.clip((ratio - 0.45) * 120.0, 0.0, 100.0)
            return float(score)
        except Exception:
            return 20.0

    def evaluate_image(
        self,
        media_name: str,
        deepfake_prob: float,
        rgb_crop: np.ndarray,
        saliency_data: Dict[str, Any]
    ) -> ForensicAssessment:
        """
        Produces forensic assessment for a single image.
        """
        prob_percent = deepfake_prob * 100.0
        freq_anomaly = self.compute_frequency_anomaly(rgb_crop)
        boundary_ratio = saliency_data.get("boundary_artifact_ratio", 0.0)

        # Multi-factor composite risk index
        # 65% Model Confidence + 20% Spatial Saliency Boundary + 15% Frequency Anomaly
        composite_risk = (
            0.65 * prob_percent +
            0.20 * np.clip(boundary_ratio * 1.5, 0.0, 100.0) +
            0.15 * freq_anomaly
        )
        composite_risk = float(np.clip(composite_risk, 0.0, 100.0))
        authenticity_score = 100.0 - composite_risk

        if composite_risk >= 75.0:
            risk_level = "CRITICAL"
            classification = "DEEPFAKE"
        elif composite_risk >= 50.0:
            risk_level = "HIGH"
            classification = "DEEPFAKE"
        elif composite_risk >= 30.0:
            risk_level = "MODERATE"
            classification = "SUSPICIOUS / INCONCLUSIVE"
        else:
            risk_level = "LOW"
            classification = "REAL"

        # Determine manipulation technique heuristic
        if classification == "REAL":
            technique = "No Synthetic Alteration Detected"
        elif boundary_ratio > 35.0:
            technique = "Face Swap / Seam Splice"
        elif "Mouth / Lips & Mandibular Boundary" in saliency_data.get("suspicious_zones", []):
            technique = "Facial Reenactment / Lip Sync"
        elif freq_anomaly > 60.0:
            technique = "Generative GAN / Diffusion Synthesis"
        else:
            technique = "Digital Facial Manipulation"

        summary = (
            f"TruthVision analyzed visual artifacts on '{media_name}'. "
            f"The neural model indicates a {prob_percent:.1f}% synthetic likelihood. "
            f"Spatial Grad-CAM inspection revealed notable cues in: {', '.join(saliency_data.get('suspicious_zones', ['general facial region']))}."
        )

        return ForensicAssessment(
            media_name=media_name,
            media_type="IMAGE",
            classification=classification,
            deepfake_probability=prob_percent,
            authenticity_score=authenticity_score,
            risk_level=risk_level,
            risk_score=composite_risk,
            likely_technique=technique,
            suspicious_regions=saliency_data.get("suspicious_zones", []),
            boundary_artifact_ratio=boundary_ratio,
            frequency_anomaly_score=freq_anomaly,
            temporal_variance=0.0,
            frames_analyzed=1,
            flagged_frames_count=1 if classification == "DEEPFAKE" else 0,
            summary_notes=summary
        )

    def evaluate_video(
        self,
        media_name: str,
        frame_probabilities: List[float],
        timestamps: List[float],
        avg_boundary_ratio: float = 0.0,
        avg_freq_anomaly: float = 0.0
    ) -> ForensicAssessment:
        """
        Produces forensic assessment for video timeline analysis.
        """
        if not frame_probabilities:
            frame_probabilities = [0.0]
            timestamps = [0.0]

        probs_arr = np.array(frame_probabilities)
        mean_prob = float(np.mean(probs_arr)) * 100.0
        peak_prob = float(np.max(probs_arr)) * 100.0
        temporal_std = float(np.std(probs_arr))

        # Flagged frames (threshold 0.5)
        flagged_indices = [i for i, p in enumerate(frame_probabilities) if p >= self.fake_threshold]
        flagged_count = len(flagged_indices)
        flagged_ratio = flagged_count / max(1, len(frame_probabilities))

        # Identify timestamp intervals of manipulation
        flagged_intervals = []
        if flagged_indices:
            start_t = None
            prev_t = None
            for idx in flagged_indices:
                t = timestamps[idx] if idx < len(timestamps) else idx * 0.5
                if start_t is None:
                    start_t = t
                elif prev_t is not None and (t - prev_t) > 1.5:  # gap detected
                    flagged_intervals.append(f"{self._format_sec(start_t)} - {self._format_sec(prev_t)}")
                    start_t = t
                prev_t = t
            if start_t is not None and prev_t is not None:
                flagged_intervals.append(f"{self._format_sec(start_t)} - {self._format_sec(prev_t)}")

        # Risk calculation incorporating temporal instability & peak frames
        composite_risk = (
            0.50 * mean_prob +
            0.25 * peak_prob +
            0.15 * (flagged_ratio * 100.0) +
            0.10 * np.clip(temporal_std * 200.0, 0.0, 100.0)
        )
        composite_risk = float(np.clip(composite_risk, 0.0, 100.0))
        authenticity_score = 100.0 - composite_risk

        if composite_risk >= 70.0 or flagged_ratio > 0.40:
            risk_level = "CRITICAL"
            classification = "DEEPFAKE"
        elif composite_risk >= 45.0 or flagged_count > 0:
            risk_level = "HIGH"
            classification = "DEEPFAKE"
        elif composite_risk >= 25.0:
            risk_level = "MODERATE"
            classification = "SUSPICIOUS / INCONCLUSIVE"
        else:
            risk_level = "LOW"
            classification = "REAL"

        if classification == "REAL":
            technique = "No Temporal or Visual Manipulation Detected"
        elif temporal_std > 0.20:
            technique = "Selective Frame Manipulation / Face Swap"
        else:
            technique = "Persistent Deepfake Reenactment"

        summary = (
            f"TruthVision inspected {len(frame_probabilities)} sample frames across '{media_name}'. "
            f"{flagged_count} frame(s) exhibited synthetic manipulation signatures. "
            f"Overall video risk is categorized as {risk_level} with a {mean_prob:.1f}% mean synthetic index."
        )

        return ForensicAssessment(
            media_name=media_name,
            media_type="VIDEO",
            classification=classification,
            deepfake_probability=mean_prob,
            authenticity_score=authenticity_score,
            risk_level=risk_level,
            risk_score=composite_risk,
            likely_technique=technique,
            suspicious_regions=["Temporal Frame Inconsistencies" if temporal_std > 0.15 else "Facial Core"],
            boundary_artifact_ratio=avg_boundary_ratio,
            frequency_anomaly_score=avg_freq_anomaly,
            temporal_variance=temporal_std,
            frames_analyzed=len(frame_probabilities),
            flagged_frames_count=flagged_count,
            flagged_intervals=flagged_intervals if flagged_intervals else ["None detected"],
            summary_notes=summary
        )

    def _format_sec(self, seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
