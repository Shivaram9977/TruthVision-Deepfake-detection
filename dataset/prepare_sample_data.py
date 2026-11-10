import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_synthetic_sample_dataset(base_dir: str = "dataset"):
    """
    Creates a sample training, validation, and test dataset with realistic face textures
    (real vs manipulated) to verify the pipeline immediately.
    """
    splits = ["train", "validation", "test"]
    categories = ["real", "fake"]

    for split in splits:
        for cat in categories:
            os.makedirs(os.path.join(base_dir, split, cat), exist_ok=True)

    def generate_face_image(is_fake: bool, seed: int) -> np.ndarray:
        np.random.seed(seed)
        img = np.zeros((256, 256, 3), dtype=np.uint8)

        # Base skin tone
        skin_tone = [
            np.random.randint(180, 240),
            np.random.randint(140, 200),
            np.random.randint(120, 180)
        ]
        img[:, :] = [np.random.randint(40, 70), np.random.randint(40, 70), np.random.randint(50, 90)]

        # Draw head / face oval
        cv2.ellipse(img, (128, 135), (75, 95), 0, 0, 360, skin_tone, -1)

        # Draw Eyes
        eye_color = (255, 255, 255)
        cv2.circle(img, (98, 115), 14, eye_color, -1)
        cv2.circle(img, (158, 115), 14, eye_color, -1)
        iris_color = (np.random.randint(30, 80), np.random.randint(60, 120), np.random.randint(20, 60))
        cv2.circle(img, (98, 115), 6, iris_color, -1)
        cv2.circle(img, (158, 115), 6, iris_color, -1)

        # Draw Nose
        cv2.line(img, (128, 125), (124, 150), (skin_tone[0]-25, skin_tone[1]-25, skin_tone[2]-25), 2)
        cv2.line(img, (124, 150), (132, 150), (skin_tone[0]-25, skin_tone[1]-25, skin_tone[2]-25), 2)

        # Draw Mouth
        cv2.ellipse(img, (128, 175), (26, 12), 0, 0, 360, (60, 60, 190), -1)

        if is_fake:
            # Inject deepfake-like boundary artifacts, color mismatch, or blurred seam
            choice = np.random.choice(["swap_seam", "blur_mouth", "color_warp", "noise"])
            if choice == "swap_seam":
                # Blending seam around face
                cv2.ellipse(img, (128, 135), (65, 85), 0, 0, 360, (skin_tone[0]+30, skin_tone[1]-20, skin_tone[2]-20), 3)
            elif choice == "blur_mouth":
                # Blurry reenacted mouth
                sub = img[155:195, 95:160]
                img[155:195, 95:160] = cv2.GaussianBlur(sub, (15, 15), 5)
            elif choice == "color_warp":
                # Color temperature mismatch inside face region
                sub = img[80:200, 70:185].astype(np.float32)
                sub[:, :, 0] = np.clip(sub[:, :, 0] * 1.3, 0, 255)
                sub[:, :, 2] = np.clip(sub[:, :, 2] * 0.7, 0, 255)
                img[80:200, 70:185] = sub.astype(np.uint8)
            else:
                # High frequency noise
                noise = np.random.normal(0, 25, img.shape).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    print("[+] Generating sample benchmark images...")
    counts = {"train": (20, 20), "validation": (6, 6), "test": (6, 6)}

    seed = 42
    for split, (n_real, n_fake) in counts.items():
        for i in range(n_real):
            img = generate_face_image(is_fake=False, seed=seed)
            cv2.imwrite(os.path.join(base_dir, split, "real", f"real_{i+1:03d}.jpg"), img)
            seed += 1
        for i in range(n_fake):
            img = generate_face_image(is_fake=True, seed=seed)
            cv2.imwrite(os.path.join(base_dir, split, "fake", f"fake_{i+1:03d}.jpg"), img)
            seed += 1

    # Also create a sample test video
    demo_video_path = os.path.join(base_dir, "sample_test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(demo_video_path, fourcc, 15.0, (256, 256))
    for frame_num in range(45):
        # First 15 frames real, next 20 frames fake, last 10 frames real
        is_f = (15 <= frame_num < 35)
        f_img = generate_face_image(is_fake=is_f, seed=100 + frame_num)
        out.write(f_img)
    out.release()

    print(f"[OK] Sample dataset created in '{base_dir}/' with demo video at '{demo_video_path}'")


if __name__ == "__main__":
    create_synthetic_sample_dataset()
