"""Central configuration for the scalp analyzer."""
from pathlib import Path

ROOT         = Path(__file__).resolve().parent
DATA_DIR     = ROOT / "data"
MODELS_DIR   = ROOT / "models"
OUTPUTS_DIR  = ROOT / "outputs"
SAMPLES_DIR  = ROOT / "samples"

CALIBRATION_FILE = DATA_DIR / "calibration_reference.json"
FOLLICLE_MODEL   = MODELS_DIR / "follicle_detector.onnx"
SHAFT_MODEL      = MODELS_DIR / "shaft_segmenter.onnx"
CLASSIFIER_MODEL = MODELS_DIR / "alopecia_classifier.onnx"

DEFAULT_MAGNIFICATION = "x20"
DEFAULT_PX_PER_MM     = 1180.0

# Provider preference order. Whatever the installed onnxruntime exposes first
# is used; unknown entries are dropped. MIGraphX is the current AMD ROCm path;
# ROCMExecutionProvider is retained for older builds.
PROVIDER_PREFERENCE = [
    "MIGraphXExecutionProvider",
    "ROCMExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]
