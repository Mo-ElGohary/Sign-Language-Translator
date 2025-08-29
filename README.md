# ASL Alphabet Sign Language Translator

A complete PyTorch project to train and run an ASL alphabet translator. It recognizes static ASL alphabet hand signs from images or a webcam and outputs predicted letters.

Note: ASL letters J and Z are dynamic (motion). This model focuses on static handshapes. If your dataset includes J and Z as static approximations, they will be treated as such.

---

## Features
- Transfer learning with MobileNetV3 (small) for fast, accurate classification
- Robust training loop with mixed precision, early stopping, and checkpointing
- Image augmentations for better generalization
- Webcam inference with optional MediaPipe Hands cropping
- Works with any dataset in ImageFolder format

---

## Project Structure
`
.
+-- README.md
+-- requirements.txt
+-- .gitignore
+-- data/
�   +-- (place datasets here)
+-- checkpoints/
�   +-- (models saved here)
+-- src/
    +-- config.py
    +-- data.py
    +-- model.py
    +-- train.py
    +-- infer.py
    +-- utils.py
`

---

## Setup
1. Create and activate a virtual environment (Windows PowerShell):
`powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
`

2. Install dependencies:
`powershell
pip install --upgrade pip
pip install -r requirements.txt
`

---

## Dataset
This project uses any dataset organized in PyTorch ImageFolder layout:
`
data/ASL_Alphabet/
  train/
    A/
      img1.jpg
      img2.jpg
      ...
    B/
    ...
  val/
    A/
    B/
    ...
`

- If you have only one folder with subfolders per class, use the split utility in the training script (--auto-split) to create 	rain/al splits.
- Popular dataset: " ASL Alphabet\ (29 classes: 26 letters + del, 
othing, space). Download it manually and arrange as above.

---

## Training
Minimal example:
`powershell
python -m src.train --data-dir data/ASL_Alphabet --epochs 10 --batch-size 64 --img-size 224 --checkpoint-dir checkpoints
`

Common options:
- --data-dir: Path containing rain/ and optionally al/ (or a single folder to auto-split)
- --epochs: Number of epochs
- --batch-size: Global batch size
- --img-size: Input image size (default 224)
- --lr: Learning rate (default 3e-4)
- --weight-decay: Weight decay (default 0.05)
- --workers: Data loader workers (default 4)
- --auto-split: If set and only one folder is provided, split into train/val
- --val-split: Ratio for validation split when auto-splitting (default 0.15)
- --use-mp-hands: Use MediaPipe Hands pre-cropping during training (slower, sometimes better)

Outputs:
- Best model weights at checkpoints/best.pt
- Last epoch weights at checkpoints/last.pt
- checkpoints/labels.json with class names

---

## Inference (Webcam)
After training and having checkpoints/best.pt:
`powershell
python -m src.infer --weights checkpoints/best.pt --use-mp-hands --camera 0
`

Options:
- --weights: Path to a saved checkpoint
- --use-mp-hands: Enable MediaPipe Hands for better cropping
- --camera: Camera index (default 0)
- --conf: Confidence threshold for displaying predictions (default 0.5)

Press Q to quit the webcam window.

---

## Inference (Image/Folder)
`powershell
python -m src.infer --weights checkpoints/best.pt --image-path path\\to\\image_or_folder --use-mp-hands
`
- If image-path is a folder, all images inside it will be processed.

---

## Tips
- Ensure good lighting and a consistent background when collecting or using images.
- For best results, use a dataset that closely matches your deployment conditions.
- If your dataset includes J and Z as dynamic signs, consider excluding them or treating them as separate labels according to your dataset definition.

---

## License
MIT
