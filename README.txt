Linux:
  chmod +x linux/*.sh
  ./linux/install.sh          # auto-detects ROCm, installs onnxruntime-rocm
  source .venv/bin/activate
  python check_env.py         # confirm MIGraphXExecutionProvider is present
  ./linux/run_analysis.sh samples/your_photo.jpg x20

Windows:
  windows\install_windows.bat  # double-click; auto-detects GPU (DirectML) or CPU
  windows\run_analysis.bat C:\path\to\your_photo.jpg x20
