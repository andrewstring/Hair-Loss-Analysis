"""Trichoscopic image standardisation and calibration."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from config import CALIBRATION_FILE, DEFAULT_PX_PER_MM


class TrichoscopicIngestor:
    """Load, white-balance, flat-field, and circle-mask a trichoscopic image."""

    def __init__(self, target_magnification: str = "x20"):
        self.target = target_magnification
        self.px_per_mm = self._load_calibration(target_magnification)

    def _load_calibration(self, mag: str) -> float:
        if CALIBRATION_FILE.exists():
            with open(CALIBRATION_FILE) as f:
                data = json.load(f)
            entry = data.get(mag)
            if isinstance(entry, dict) and "px_per_mm" in entry:
                return float(entry["px_per_mm"])
            if isinstance(entry, (int, float)):
                return float(entry)
        return DEFAULT_PX_PER_MM

    def standardise(self, image_path: str | Path) -> dict:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected 3-channel BGR image, got shape {img.shape}")

        # 1. Illumination flattening (CLAHE on the L channel)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        flat = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

        # 2. Grey-world white balance
        flat = self._grey_world_balance(flat)

        # 3. Circular crop to the illuminated field
        h, w = flat.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        radius = min(h, w) * 0.46
        circle_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        flat = np.where(circle_mask[..., None], flat, 0).astype(np.uint8)

        area_mm2 = float(np.pi * (radius / self.px_per_mm) ** 2)

        return {
            "image": flat,
            "px_per_mm": self.px_per_mm,
            "area_mm2": area_mm2,
            "radius_px": float(radius),
            "magnification": self.target,
        }

    @staticmethod
    def _grey_world_balance(img: np.ndarray) -> np.ndarray:
        result = img.astype(np.float32)
        avg = result.reshape(-1, 3).mean(axis=0)
        gray = float(avg.mean())
        scale = gray / (avg + 1e-6)
        result *= scale
        return np.clip(result, 0, 255).astype(np.uint8)
