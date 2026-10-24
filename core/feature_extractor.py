"""Follicle detection, shaft-width measurement, and trichoscopic profile extraction.

Uses ONNX models when present under models/; otherwise falls back to OpenCV
heuristics that require no weights. ONNX sessions prefer the MIGraphX provider
on AMD ROCm systems, falling back to ROCM / CUDA / CPU as available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from config import FOLLICLE_MODEL, SHAFT_MODEL
from core import runtime


@dataclass
class FollicleObservation:
    bbox: tuple
    shaft_width_px: float
    shaft_width_um: float
    is_vellus: bool
    confidence: float


@dataclass
class TrichoscopicProfile:
    follicle_count: int
    density_per_cm2: float
    mean_shaft_width_um: float
    std_shaft_width_um: float
    diameter_diversity_pct: float
    vellus_ratio_pct: float
    terminal_to_vellus_ratio: float
    single_hair_unit_ratio: float
    empty_follicle_ratio: float
    exclamation_hair_count: int
    black_dot_count: int
    yellow_dot_count: int
    perifollicular_pigmentation_score: float
    ostia_visibility_score: float
    observations: List[FollicleObservation] = field(default_factory=list)
    used_models: dict = field(default_factory=dict)


class FeatureExtractor:
    VELLUS_THRESHOLD_UM = 30.0
    TERMINAL_THRESHOLD_UM = 60.0

    def __init__(self,
                 follicle_model_path=FOLLICLE_MODEL,
                 shaft_model_path=SHAFT_MODEL):
        self.follicle_sess = runtime.make_session(follicle_model_path)
        self.shaft_sess    = runtime.make_session(shaft_model_path)

    # ------------------------------------------------------------------ main
    def extract(self, standardised: dict, px_per_mm: Optional[float] = None) -> TrichoscopicProfile:
        img = standardised["image"]
        if px_per_mm is None:
            px_per_mm = standardised["px_per_mm"]
        px_per_um = px_per_mm / 1000.0
        area_mm2 = standardised["area_mm2"]

        used_models = {
            "follicle_detector": self.follicle_sess is not None,
            "shaft_segmenter":   self.shaft_sess is not None,
            "device":            runtime.describe_device(),
        }

        if self.follicle_sess is not None:
            boxes = self._detect_follicles_onnx(img)
        else:
            boxes = self._detect_follicles_heuristic(img)

        if not boxes:
            raise RuntimeError("No follicles detected; check image quality / calibration.")

        shaft_mask = self._shaft_mask(img)

        if self.shaft_sess is not None:
            observations = [self._obs_from_onnx(img, b, c, px_per_um) for b, c in boxes]
        else:
            observations = [self._obs_from_heuristic(img, shaft_mask, b, c, px_per_um)
                            for b, c in boxes]

        widths = np.array([o.shaft_width_um for o in observations if o.shaft_width_um > 0])
        if widths.size == 0:
            widths = np.array([self.VELLUS_THRESHOLD_UM])

        mean_w = float(widths.mean())
        std_w  = float(widths.std())
        diversity_cv = float((std_w / (mean_w + 1e-6)) * 100.0)

        vellus   = [o for o in observations if o.is_vellus]
        terminal = [o for o in observations if not o.is_vellus]
        density_per_cm2 = len(observations) / max(area_mm2 / 100.0, 1e-6)

        return TrichoscopicProfile(
            follicle_count=len(observations),
            density_per_cm2=density_per_cm2,
            mean_shaft_width_um=mean_w,
            std_shaft_width_um=std_w,
            diameter_diversity_pct=diversity_cv,
            vellus_ratio_pct=(len(vellus) / len(observations)) * 100.0,
            terminal_to_vellus_ratio=len(terminal) / max(len(vellus), 1),
            single_hair_unit_ratio=self._single_unit_ratio(observations),
            empty_follicle_ratio=self._empty_follicle_ratio(shaft_mask, observations),
            exclamation_hair_count=self._count_exclamation_hairs(shaft_mask),
            black_dot_count=self._count_black_dots(img),
            yellow_dot_count=self._count_yellow_dots(img),
            perifollicular_pigmentation_score=self._pigmentation_score(img, observations),
            ostia_visibility_score=self._ostia_score(img),
            observations=observations,
            used_models=used_models,
        )

    # -------------------------------------------------------------- detectors
    def _detect_follicles_onnx(self, img: np.ndarray):
        blob = self._preprocess_yolo(img)
        dets = self.follicle_sess.run(None, {"images": blob})[0]
        return self._nms(dets, conf_thres=0.35, iou_thres=0.45)

    @staticmethod
    def _detect_follicles_heuristic(img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(th, connectivity=8)

        boxes = []
        h_img, w_img = gray.shape
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            if not (8 <= area <= 400):
                continue
            if w == 0 or h == 0:
                continue
            if not (0.4 <= (w / h) <= 2.5):
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_img, x + w), min(h_img, y + h)
            circularity = 4.0 * np.pi * area / (w + h) ** 2
            conf = float(min(1.0, max(0.15, circularity)))
            boxes.append(((x1, y1, x2, y2), conf))
        return boxes

    @staticmethod
    def _shaft_mask(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
        _, mask = cv2.threshold(blackhat, 18, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        return mask

    def _obs_from_onnx(self, img, box, conf, px_per_um):
        x1, y1, x2, y2 = box
        roi = img[y1:y2, x1:x2]
        width_px = self._shaft_width_onnx(roi) if roi.size else 0.0
        return self._make_obs(box, conf, width_px, px_per_um)

    def _obs_from_heuristic(self, img, shaft_mask, box, conf, px_per_um):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r = 40
        h, w = shaft_mask.shape
        y_lo, y_hi = max(0, cy - r), min(h, cy + r)
        x_lo, x_hi = max(0, cx - r), min(w, cx + r)
        region = shaft_mask[y_lo:y_hi, x_lo:x_hi]
        if region.size == 0 or not region.any():
            width_px = 0.0
        else:
            dist = cv2.distanceTransform(region, cv2.DIST_L2, 5)
            width_px = float(dist.max() * 2.0)
        return self._make_obs(box, conf, width_px, px_per_um)

    def _make_obs(self, box, conf, width_px, px_per_um):
        width_um = width_px * px_per_um
        return FollicleObservation(
            bbox=tuple(int(v) for v in box),
            shaft_width_px=float(width_px),
            shaft_width_um=float(width_um),
            is_vellus=width_um < self.VELLUS_THRESHOLD_UM,
            confidence=float(conf),
        )

    def _shaft_width_onnx(self, roi) -> float:
        blob = cv2.resize(roi, (128, 128)).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        out = self.shaft_sess.run(None, {"input": blob})[0]
        mask = (out[0, 0] > 0.5).astype(np.uint8)
        if mask.sum() < 10:
            return 0.0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        widths = []
        for c in contours:
            area = cv2.contourArea(c)
            perim = cv2.arcLength(c, True)
            if perim > 0:
                widths.append(2.0 * area / perim)
        return float(np.median(widths)) if widths else 0.0

    # ------------------------------------------------------------- aggregates
    def _single_unit_ratio(self, obs: List[FollicleObservation]) -> float:
        if not obs:
            return 0.0
        singles = sum(1 for o in obs if o.shaft_width_um < self.TERMINAL_THRESHOLD_UM)
        return singles / len(obs) * 100.0

    @staticmethod
    def _empty_follicle_ratio(shaft_mask: np.ndarray, obs: List[FollicleObservation]) -> float:
        if not obs:
            return 0.0
        h, w = shaft_mask.shape
        empty = 0
        for o in obs:
            x1, y1, x2, y2 = o.bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            r = 15
            y_lo, y_hi = max(0, cy - r), min(h, cy + r)
            x_lo, x_hi = max(0, cx - r), min(w, cx + r)
            local = shaft_mask[y_lo:y_hi, x_lo:x_hi]
            if local.size == 0 or local.sum() < 20:
                empty += 1
        return empty / len(obs)

    @staticmethod
    def _count_exclamation_hairs(shaft_mask: np.ndarray) -> int:
        contours, _ = cv2.findContours(shaft_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        for c in contours:
            area = cv2.contourArea(c)
            perim = cv2.arcLength(c, True)
            if perim < 20 or perim > 400:
                continue
            if perim > 0 and area / perim < 0.6:
                count += 1
        return count

    @staticmethod
    def _count_black_dots(img: np.ndarray) -> int:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        n, _, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
        return int(sum(1 for i in range(1, n) if 10 <= stats[i, cv2.CC_STAT_AREA] <= 40))

    @staticmethod
    def _count_yellow_dots(img: np.ndarray) -> int:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (15, 40, 80), (40, 255, 255))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        return int(sum(1 for i in range(1, n) if 5 < stats[i, cv2.CC_STAT_AREA] < 200))

    @staticmethod
    def _pigmentation_score(img: np.ndarray, obs: List[FollicleObservation]) -> float:
        if not obs:
            return 0.0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        scores = []
        for o in obs:
            x1, y1, x2, y2 = o.bbox
            pad = 6
            y_lo, y_hi = max(0, y1 - pad), min(h, y2 + pad)
            x_lo, x_hi = max(0, x1 - pad), min(w, x2 + pad)
            roi = hsv[y_lo:y_hi, x_lo:x_hi]
            if roi.size == 0:
                continue
            brown = cv2.inRange(roi, (5, 40, 40), (25, 255, 200))
            scores.append(float(brown.mean()) / 255.0)
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def _ostia_score(img: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=6,
            param1=50, param2=15, minRadius=1, maxRadius=5,
        )
        if circles is None:
            return 0.0
        return float(min(1.0, len(circles[0]) / 80.0))

    @staticmethod
    def _preprocess_yolo(img: np.ndarray) -> np.ndarray:
        resized = cv2.resize(img, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))[None, ...]

    @staticmethod
    def _nms(dets: np.ndarray, conf_thres: float, iou_thres: float):
        d = dets[0]
        keep = d[:, 4] > conf_thres
        d = d[keep]
        if len(d) == 0:
            return []
        boxes = d[:, :4].tolist()
        scores = d[:, 4].tolist()
        idx = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
        if len(idx) == 0:
            return []
        idx = np.array(idx).flatten()
        return list(zip([tuple(int(v) for v in boxes[i]) for i in idx],
                        [float(scores[i]) for i in idx]))
