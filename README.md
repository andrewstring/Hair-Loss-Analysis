# Scalp Analyzer — Trichoscopic AGA Profiling & Aggressive Treatment Planner

Trichoscopic image → follicle metrics → differential diagnosis (AGA / AA / TE /
scarring) → aggressive treatment protocol including research chemicals.

Works without trained models. Default pipeline uses OpenCV heuristics + a
literature-grounded rule classifier. Drop ONNX models into `models/` to upgrade.

## Install (Ubuntu / Linux)

    chmod +x install.sh && ./install.sh

The installer auto-detects an AMD ROCm stack and installs `onnxruntime-rocm`.
Force behaviour:

    INSTALL_MODE=cpu  ./install.sh     # CPU only
    INSTALL_MODE=rocm ./install.sh     # Force ROCm wheel
    INSTALL_MODE=auto ./install.sh     # Default: detect

## Verify GPU is active

    python check_env.py

Expect `MIGraphXExecutionProvider` in the provider list on AMD ROCm systems,
or `CUDAExecutionProvider` on NVIDIA. GPU inference only runs when the
corresponding `.onnx` models also exist under `models/`.

## Run

    source .venv/bin/activate
    python main.py /path/to/scalp.jpg
    python main.py /path/to/scalp.jpg --mag x50 --json outputs/report.json

## Safety

Engineering artefact. Not a medical device. The treatment planner is
deliberately aggressive per user request and includes unapproved research
chemicals. Physician review required before any agent is used.
