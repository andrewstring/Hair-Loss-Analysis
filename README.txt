chmod +x install.sh
./install.sh                # auto-detects ROCm, installs onnxruntime-rocm
source .venv/bin/activate
python check_env.py         # confirm MIGraphXExecutionProvider is present
python main.py samples/your_photo.jpg --mag x20 --json outputs/report.json
