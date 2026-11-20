## Model & Training

- **Model:** YOLOv8m (Ultralytics), fine-tuned from COCO-pretrained weights (transfer learning)
- **Task:** Object detection, two classes: `helmet` / `no_helmet`
- **Framework:** Ultralytics YOLO, PyTorch, CUDA GPU
- **Image size:** 640×640
- **Batch size:** 4
- **Epochs:** <EPOCHS from config.py>
- **Initial learning rate:** 0.005

### Augmentation
Default Ultralytics augmentations (mosaic, HSV shift, horizontal flip, scale, translation) plus:
- Rotation: ±10°
- Copy-paste: 0.3
- MixUp: 0.1

### Pipeline
1. Organize dataset into train/valid splits (YOLO format)
2. Generate dataset YAML config
3. Fine-tune YOLOv8m
4. Validate each epoch (mAP, precision, recall) and keep `best.pt`
5. Export best weights to ONNX

### Outputs
Loss/mAP curves, confusion matrix, PR/F1 curves, validation batches
with custom green (helmet) / red (no helmet) boxes, `best.pt`, and ONNX export.
