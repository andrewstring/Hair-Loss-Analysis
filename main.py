"""Scalp Analyzer CLI."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from core.image_ingest import TrichoscopicIngestor
from core.feature_extractor import FeatureExtractor
from core.alopecia_classifier import AlopeciaClassifier
from core.treatment_planner import AggressiveTreatmentPlanner
from core import runtime


def analyze(image_path: str, magnification: str):
    ingestor = TrichoscopicIngestor(target_magnification=magnification)
    standardised = ingestor.standardise(image_path)
    extractor = FeatureExtractor()
    profile = extractor.extract(standardised, px_per_mm=standardised["px_per_mm"])
    classifier = AlopeciaClassifier()
    diagnosis = classifier.classify(profile)
    planner = AggressiveTreatmentPlanner()
    plan = planner.plan(diagnosis, profile)
    return standardised, profile, diagnosis, plan


def _profile_to_dict(p) -> dict:
    d = asdict(p)
    d["observations"] = [
        {"shaft_width_um": round(o.shaft_width_um, 2),
         "is_vellus": o.is_vellus,
         "confidence": round(o.confidence, 3)}
        for o in p.observations
    ]
    return d


def print_report(standardised, profile, diagnosis, plan):
    print("\n" + "=" * 66)
    print("  SCALP ANALYZER - TRICHOSCOPIC REPORT")
    print("=" * 66)
    print(f"  Magnification:      {standardised['magnification']}")
    print(f"  px_per_mm:          {standardised['px_per_mm']:.1f}")
    print(f"  Field area (mm^2):  {standardised['area_mm2']:.2f}")
    print(f"  Runtime device:     {runtime.describe_device()}")
    print(f"  Models in use:      {profile.used_models}")

    print("\n--- FEATURE SUMMARY ---")
    print(f"  Follicles detected:        {profile.follicle_count}")
    print(f"  Density (per cm^2):        {profile.density_per_cm2:.1f}")
    print(f"  Mean shaft width (um):     {profile.mean_shaft_width_um:.1f}")
    print(f"  Std shaft width (um):      {profile.std_shaft_width_um:.1f}")
    print(f"  Diameter diversity (%CV):  {profile.diameter_diversity_pct:.1f}")
    print(f"  Vellus ratio (%):          {profile.vellus_ratio_pct:.1f}")
    print(f"  Terminal:vellus ratio:     {profile.terminal_to_vellus_ratio:.2f}")
    print(f"  Single-hair unit ratio:    {profile.single_hair_unit_ratio:.1f}%")
    print(f"  Empty follicle ratio:      {profile.empty_follicle_ratio:.1%}")
    print(f"  Exclamation hairs:         {profile.exclamation_hair_count}")
    print(f"  Black dots:                {profile.black_dot_count}")
    print(f"  Yellow dots:               {profile.yellow_dot_count}")
    print(f"  Perifollicular pigment:    {profile.perifollicular_pigmentation_score:.2f}")
    print(f"  Ostia visibility:          {profile.ostia_visibility_score:.2f}")

    print("\n--- DIFFERENTIAL DIAGNOSIS ---")
    print(f"  PRIMARY: {diagnosis.primary}  (confidence {diagnosis.confidence})")
    print(f"  AGA severity hint: {diagnosis.aga_severity_hint}")
    print("  Probabilities:")
    for k, v in sorted(diagnosis.probabilities.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(round(v * 30))
        print(f"    {k:<10} {v:6.3f}  {bar}")
    print("  Rationale:")
    for r in diagnosis.rationale:
        print(f"    - {r}")
    if diagnosis.rule_out_notes:
        print("  Rule-outs:")
        for n in diagnosis.rule_out_notes:
            print(f"    - {n}")

    print("\n--- TREATMENT PLAN ---")
    print(f"  {plan.diagnosis_summary}")
    for i, c in enumerate(plan.components, 1):
        print(f"\n  [{i}] {c.category} - {c.agent}")
        print(f"       Protocol:  {c.dose_or_protocol}")
        print(f"       Evidence:  {c.evidence_tier}")
        print(f"       Rationale: {c.rationale}")
        print(f"       Risks:     {c.risk_notes}")
    print("\n  Monitoring:")
    for m in plan.monitoring:
        print(f"    - {m}")
    print("\n  Escalation logic:")
    for e in plan.escalation_logic:
        print(f"    - {e}")
    print("\n  " + "-" * 62)
    print("  " + plan.full_disclosure)
    print("=" * 66 + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Trichoscopic scalp analysis + aggressive treatment planner.")
    ap.add_argument("image", help="Path to a trichoscopic scalp image.")
    ap.add_argument("--mag", default="x20", choices=["x20", "x50", "x100"])
    ap.add_argument("--json", metavar="PATH", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    try:
        standardised, profile, diagnosis, plan = analyze(args.image, args.mag)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print_report(standardised, profile, diagnosis, plan)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input": str(args.image),
            "magnification": standardised["magnification"],
            "px_per_mm": standardised["px_per_mm"],
            "area_mm2": standardised["area_mm2"],
            "runtime_device": runtime.describe_device(),
            "profile": _profile_to_dict(profile),
            "diagnosis": asdict(diagnosis),
            "plan": plan.to_dict(),
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote report -> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
