# Scalp Analyzer — Trichoscopic AGA Profiling & Aggressive Treatment Planner

Trichoscopic image → follicle metrics → differential diagnosis (AGA / AA / TE /
scarring) → aggressive treatment protocol including research chemicals.

Works without trained models. Default pipeline uses OpenCV heuristics + a
literature-grounded rule classifier. Drop ONNX models into `models/` to upgrade.

## Install (Ubuntu / Linux)

    chmod +x linux/*.sh && ./linux/install.sh

The installer auto-detects an AMD ROCm stack and installs `onnxruntime-rocm`.
Force behaviour:

    INSTALL_MODE=cpu  ./linux/install.sh     # CPU only
    INSTALL_MODE=rocm ./linux/install.sh     # Force ROCm wheel
    INSTALL_MODE=auto ./linux/install.sh     # Default: detect

## Install (Windows)

Double-click `windows\install_windows.bat` (or run it from a terminal). It installs
Python 3.12 via winget if needed, creates `.venv`, installs dependencies and picks
the right ONNX Runtime build (DirectML for GPUs, otherwise CPU).

```
windows\install_windows.bat                # auto-detect
windows\install_windows.bat -Mode cpu      # CPU only
windows\install_windows.bat -Mode directml # any DX12 GPU
windows\install_windows.bat -Mode cuda     # NVIDIA, needs CUDA + cuDNN
windows\install_windows.bat -Recreate      # rebuild .venv
```

Then drag an image onto `windows\run_analysis.bat`, or run
`windows\run_analysis.bat C:\path\to\scalp.jpg x20`.

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
