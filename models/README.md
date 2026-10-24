# Optional ONNX models

Not required. If present, the pipeline auto-upgrades to model-based inference
and will run on MIGraphX (AMD ROCm), CUDA, or CPU depending on the installed
`onnxruntime` build.

| File                       | Stage      | Recipe |
|----------------------------|------------|--------|
| `follicle_detector.onnx`   | Detection  | YOLOv8 fine-tuned on trichoscopic follicle annotations. mAP@0.5 ~ 0.92, recall ~ 0.86. Input tensor name must be `images`, output shape (1, N, 5+). |
| `shaft_segmenter.onnx`     | Width      | U-Net / EfficientNet segmentation head. Input tensor `input` (N, 3, 128, 128) normalised to [0,1], output (N, 1, 128, 128) mask. |
| `alopecia_classifier.onnx` | Diagnosis  | MLP on the 11-dim profile vector. Input `features` (N, 11). Output logits (N, 4) for classes [AGA, AA, TE, Scarring]. |

## Exporting YOLOv8 for AMD

    from ultralytics import YOLO
    model = YOLO("runs/detect/train/weights/best.pt")
    model.export(format="onnx", opset=17, simplify=True, dynamic=False)

Ultralytics' ONNX backend auto-selects MIGraphXExecutionProvider on ROCm hosts,
so the exported model runs on your Radeon GPU without code changes.

## Notes

- MIGraphX fuses ops aggressively and may reject unusual ops; those fall through
  to ROCMExecutionProvider or CPUExecutionProvider at the op level. This is
  expected and produces correct results.
- If a model silently runs slower than expected, set:

      export ONNXRUNTIME_EP=CPUExecutionProvider

  to force CPU for A/B comparison.
