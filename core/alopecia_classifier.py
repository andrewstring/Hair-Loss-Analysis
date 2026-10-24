"""Differential diagnosis: AGA vs AA vs TE vs scarring.

Blends an optional ONNX MLP with a literature-grounded rule layer. Rules drive
the diagnosis when no model is present. The ONNX session uses the AMD MIGraphX
provider when onnxruntime-rocm is installed, otherwise CPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import CLASSIFIER_MODEL
from core import runtime
from core.feature_extractor import TrichoscopicProfile


CLASSES = ["AGA", "AA", "TE", "Scarring"]


@dataclass
class AlopeciaDiagnosis:
    primary: str
    probabilities: dict
    confidence: str
    rationale: list
    aga_severity_hint: str
    rule_out_notes: list
    used_model: bool


class AlopeciaClassifier:
    def __init__(self, model_path=CLASSIFIER_MODEL):
        self.sess = runtime.make_session(model_path)

    def classify(self, profile: TrichoscopicProfile) -> AlopeciaDiagnosis:
        model_probs = self._model_probabilities(profile) if self.sess else None
        rule_scores, reasons = self._rule_scores(profile)
        blended = self._blend(model_probs, rule_scores)
        primary = max(blended, key=blended.get)
        confidence = self._confidence(blended, reasons)

        return AlopeciaDiagnosis(
            primary=primary,
            probabilities=blended,
            confidence=confidence,
            rationale=reasons,
            aga_severity_hint=self._aga_severity(profile),
            rule_out_notes=self._rule_out_notes(profile, blended),
            used_model=self.sess is not None,
        )

    def _model_probabilities(self, p: TrichoscopicProfile) -> dict:
        vec = np.array([[
            p.density_per_cm2,
            p.mean_shaft_width_um,
            p.diameter_diversity_pct,
            p.vellus_ratio_pct,
            p.single_hair_unit_ratio,
            p.empty_follicle_ratio,
            p.black_dot_count / 10.0,
            p.yellow_dot_count / 10.0,
            p.exclamation_hair_count / 5.0,
            p.perifollicular_pigmentation_score,
            p.ostia_visibility_score,
        ]], dtype=np.float32)
        logits = self.sess.run(None, {"features": vec})[0][0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        return dict(zip(CLASSES, probs.tolist()))

    @staticmethod
    def _rule_scores(p: TrichoscopicProfile):
        scores = {c: 0.0 for c in CLASSES}
        reasons: list = []

        # AGA
        if p.diameter_diversity_pct > 20:
            scores["AGA"] += 3.0
            reasons.append(f"Hair shaft diameter diversity {p.diameter_diversity_pct:.1f}% > 20% - diagnostic of AGA.")
        if p.vellus_ratio_pct > 15:
            scores["AGA"] += 2.0
            reasons.append(f"Vellus/miniaturised hair ratio {p.vellus_ratio_pct:.1f}% suggests follicular miniaturisation.")
        if p.perifollicular_pigmentation_score > 0.25:
            scores["AGA"] += 1.5
            reasons.append("Perifollicular pigmentation present - supportive of AGA.")
        if p.terminal_to_vellus_ratio < 4:
            scores["AGA"] += 1.0
            reasons.append("Terminal-to-vellus ratio reduced, consistent with AGA shift.")

        # AA
        if p.exclamation_hair_count >= 2:
            scores["AA"] += 4.0
            reasons.append(f"{p.exclamation_hair_count} exclamation-mark hair(s) detected - highly specific for alopecia areata.")
        if p.black_dot_count >= 4:
            scores["AA"] += 2.5
            reasons.append(f"{p.black_dot_count} black dots - favour AA over TE/AGA.")

        # TE
        if p.empty_follicle_ratio > 0.15:
            scores["TE"] += 3.0
            reasons.append(f"Empty follicle ratio {p.empty_follicle_ratio:.1%} - supports telogen effluvium.")
        if p.yellow_dot_count >= 5 and p.black_dot_count < 3:
            scores["TE"] += 2.0
            reasons.append("Diffuse yellow dots without black dots - TE pattern.")
        if p.diameter_diversity_pct < 15 and p.vellus_ratio_pct < 10:
            scores["TE"] += 1.5
            reasons.append("Preserved shaft calibre (diversity < 15%, vellus < 10%) argues against AGA and toward TE.")

        # Scarring
        if p.ostia_visibility_score < 0.25:
            scores["Scarring"] += 4.0
            reasons.append("Follicular ostia largely absent - raises concern for scarring alopecia.")
        if p.density_per_cm2 < 40:
            scores["Scarring"] += 1.0
            reasons.append("Very low density may indicate follicular destruction; correlate with ostia score.")

        if not reasons:
            reasons.append("No decisive trichoscopic criteria fired; probability distribution reflects feature similarity only.")

        return scores, reasons

    @staticmethod
    def _blend(model_probs, rule_scores):
        keys = list(rule_scores)
        arr = np.array([rule_scores[k] for k in keys], dtype=np.float64)
        exp = np.exp(arr - arr.max())
        rule_probs = exp / exp.sum()
        rule_dict = dict(zip(keys, rule_probs.tolist()))

        if model_probs is None:
            total = sum(rule_dict.values())
            return {k: v / total for k, v in rule_dict.items()}

        blended = {k: 0.45 * model_probs.get(k, 0.0) + 0.55 * rule_dict.get(k, 0.0)
                   for k in keys}
        total = sum(blended.values())
        return {k: v / total for k, v in blended.items()}

    @staticmethod
    def _confidence(probs, reasons):
        top = max(probs.values())
        if top > 0.70 and len(reasons) >= 2:
            return "High"
        if top > 0.50:
            return "Moderate"
        return "Low"

    @staticmethod
    def _aga_severity(p: TrichoscopicProfile) -> str:
        if p.density_per_cm2 > 180 and p.diameter_diversity_pct < 30:
            return "Early / mild miniaturisation"
        if p.density_per_cm2 > 100:
            return "Moderate miniaturisation"
        return "Advanced miniaturisation / low density"

    @staticmethod
    def _rule_out_notes(p, blended):
        notes: list = []
        if blended.get("AA", 0) < 0.15 and blended.get("TE", 0) < 0.15:
            notes.append("AA and TE probabilities are low; no exclamation hairs or empty-follicle predominance.")
        if p.ostia_visibility_score >= 0.25:
            notes.append("Ostia are visible - no trichoscopic evidence of scarring alopecia.")
        if p.diameter_diversity_pct > 20 and p.vellus_ratio_pct > 15:
            notes.append("The miniaturisation signature is strongly AGA-patterned rather than diffuse TE.")
        return notes
