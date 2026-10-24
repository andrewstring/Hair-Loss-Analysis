"""ONNX Runtime provider selection and session factory (CPU / ROCm / CUDA)."""
from __future__ import annotations

import os
from typing import Optional

try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    ort = None  # type: ignore
    _HAS_ORT = False

from config import PROVIDER_PREFERENCE


def has_onnxruntime() -> bool:
    return _HAS_ORT


def available_providers() -> list[str]:
    if not _HAS_ORT:
        return []
    try:
        return list(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        return []


def get_providers() -> list[str]:
    """Return provider list filtered to those the runtime actually exposes."""
    exposed = set(available_providers())
    ordered = [p for p in PROVIDER_PREFERENCE if p in exposed]
    if "CPUExecutionProvider" not in ordered:
        ordered.append("CPUExecutionProvider")
    return ordered


def describe_device() -> str:
    """Human-readable summary of the active accelerator."""
    provs = get_providers()
    if provs and provs[0] != "CPUExecutionProvider":
        return provs[0]
    return "CPUExecutionProvider"


def make_session(path, session_options: Optional["ort.SessionOptions"] = None):
    """Create an ONNX Runtime session, or return None if unavailable."""
    if not _HAS_ORT or not os.path.exists(path):
        return None
    try:
        return ort.InferenceSession(
            str(path),
            sess_options=session_options,
            providers=get_providers(),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[runtime] Warning: cannot load {path}: {exc}")
        return None
