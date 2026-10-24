#!/usr/bin/env python3
"""Environment check — reports Python, ROCm, ONNX Runtime, providers, and models."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _try(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def rocm_info() -> tuple[bool, str]:
    version = ""
    ver_file = Path("/opt/rocm/.info/version")
    if ver_file.exists():
        try:
            version = ver_file.read_text().strip().splitlines()[0]
        except Exception:  # noqa: BLE001
            pass
    if not version and shutil.which("hipconfig"):
        version = _try(["hipconfig", "--version"]).splitlines()[0] if _try(["hipconfig", "--version"]) else ""
    present = bool(version) or Path("/opt/rocm").exists()
    return present, version


def gpu_name() -> str:
    if shutil.which("amd-smi"):
        out = _try(["amd-smi", "static"])
        for line in out.splitlines():
            if "GPU" in line and ":" in line:
                return line.strip()
    if shutil.which("rocm-smi"):
        out = _try(["rocm-smi", "--showproductname"])
        for line in out.splitlines():
            if "Card series" in line or "GPU" in line:
                return line.strip()
    return "(not detected)"


def main() -> int:
    print("=" * 66)
    print("  SCALP ANALYZER - ENVIRONMENT CHECK")
    print("=" * 66)
    print(f"  OS:           {platform.platform()}")
    print(f"  Python:       {sys.version.splitlines()[0]}")
    print(f"  Executable:   {sys.executable}")

    rocm_ok, rocm_ver = rocm_info()
    print(f"  ROCm present: {rocm_ok}  version={rocm_ver or 'unknown'}")
    if rocm_ok:
        print(f"  GPU:          {gpu_name()}")

    try:
        import numpy as np
        print(f"  numpy:        {np.__version__}")
    except ImportError:
        print("  numpy:        MISSING")

    try:
        import cv2
        print(f"  opencv:       {cv2.__version__}")
    except ImportError:
        print("  opencv:       MISSING")

    try:
        import onnxruntime as ort
        print(f"  onnxruntime:  {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"  providers:    {providers}")
        accel = [p for p in providers if p != "CPUExecutionProvider"]
        if not accel:
            print("  WARNING:      No GPU provider exposed; inference will run on CPU.")
        elif any(p.startswith(("MIGraphX", "ROCM")) for p in accel):
            print("  OK:           AMD ROCm acceleration available.")
        elif "CUDAExecutionProvider" in accel:
            print("  OK:           CUDA acceleration available.")
    except ImportError:
        print("  onnxruntime:  MISSING (pip install onnxruntime-rocm or onnxruntime)")

    print("\n--- MODELS ---")
    models = {
        "follicle_detector.onnx":   ROOT / "models" / "follicle_detector.onnx",
        "shaft_segmenter.onnx":     ROOT / "models" / "shaft_segmenter.onnx",
        "alopecia_classifier.onnx": ROOT / "models" / "alopecia_classifier.onnx",
    }
    for name, path in models.items():
        status = "present" if path.exists() else "MISSING (heuristic fallback will be used)"
        print(f"  {name:<28} {status}")

    print("\n--- RUNTIME MODE ---")
    try:
        sys.path.insert(0, str(ROOT))
        from core import runtime  # noqa: E402
        print(f"  Active device: {runtime.describe_device()}")
        print(f"  Provider list: {runtime.get_providers()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  Could not load core.runtime: {exc}")

    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
