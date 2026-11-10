import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import List, Tuple, Optional, Union


def get_default_transform(img_size: int = 224) -> transforms.Compose:
    """
    Standard PyTorch normalization transform for pretrained vision backbones.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class FaceDetector:
    """
    Robust Face Detector supporting OpenCV 4.x and 5.x with margin padding.
    """
    def __init__(self, margin: float = 0.25):
        self.margin = margin
        self.face_cascade = None
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception:
                self.face_cascade = None

    def detect_faces(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects bounding boxes [(x, y, w, h)] in a BGR OpenCV image.
        """
        if self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(40, 40)
                )
                if len(faces) > 0:
                    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
            except Exception:
                pass

        # Fallback heuristic: Center portrait region (standard for facial photos)
        h, w = image_bgr.shape[:2]
        box_w = int(w * 0.70)
        box_h = int(h * 0.75)
        x = max(0, (w - box_w) // 2)
        y = max(0, int(h * 0.10))
        return [(x, y, box_w, box_h)]

    def extract_face_crop(self, image_bgr: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extracts face crop with margin. If no face is found, returns center crop of the image.
        Returns: (face_crop_rgb, bbox_used)
        """
        h_img, w_img = image_bgr.shape[:2]

        if bbox is None:
            faces = self.detect_faces(image_bgr)
            if len(faces) > 0:
                # Select the largest face by area
                bbox = max(faces, key=lambda b: b[2] * b[3])
            else:
                # Center crop fallback
                min_dim = min(h_img, w_img)
                x = (w_img - min_dim) // 2
                y = (h_img - min_dim) // 2
                bbox = (x, y, min_dim, min_dim)

        x, y, w, h = bbox
        # Apply margin padding
        margin_w = int(w * self.margin)
        margin_h = int(h * self.margin)

        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(w_img, x + w + margin_w)
        y2 = min(h_img, y + h + margin_h)

        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            crop = image_bgr

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return crop_rgb, (x1, y1, x2 - x1, y2 - y1)


def preprocess_image_or_crop(
    image_input: Union[str, np.ndarray, Image.Image],
    img_size: int = 224,
    detector: Optional[FaceDetector] = None
) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int, int, int]]:
    """
    Helper function to load image, extract face, and convert to normalized tensor.
    Returns: (tensor [1, 3, H, W], face_crop_rgb, bbox)
    """
    if detector is None:
        detector = FaceDetector()

    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
        if img_bgr is None:
            raise ValueError(f"Could not load image from {image_input}")
    elif isinstance(image_input, Image.Image):
        img_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3 and image_input.shape[2] == 3:
            img_bgr = image_input
        else:
            raise ValueError("Input ndarray must have 3 channels.")
    else:
        raise TypeError("Unsupported image input type.")

    face_crop_rgb, bbox = detector.extract_face_crop(img_bgr)
    pil_crop = Image.fromarray(face_crop_rgb)
    transform = get_default_transform(img_size)
    tensor = transform(pil_crop).unsqueeze(0)  # Shape [1, 3, 224, 224]

    return tensor, face_crop_rgb, bbox
