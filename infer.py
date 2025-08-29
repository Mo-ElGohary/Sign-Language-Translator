#!/usr/bin/env python3
"""
ASL Alphabet Inference Script
"""

import argparse
import cv2
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.append(str(Path(__file__).parent))

from model import create_model
from utils import get_device, load_checkpoint, MediaPipeHandCropper


class ASLPredictor:
    def __init__(self, weights_path: str, img_size: int = 224, use_mediapipe_hands: bool = True, device: Optional[torch.device] = None):
        self.img_size = img_size
        self.device = device or get_device()
        self.use_mediapipe_hands = use_mediapipe_hands
        
        self.model = self._load_model(weights_path)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if use_mediapipe_hands:
            self.hand_cropper = MediaPipeHandCropper(static_image_mode=False)
    
    def _load_model(self, weights_path: str):
        print(f"Loading model from {weights_path}")
        
        checkpoint = load_checkpoint(weights_path, map_location=self.device)
        class_names = checkpoint["class_names"]
        
        print(f"Found {len(class_names)} classes: {class_names}")
        
        model = create_model(num_classes=len(class_names), pretrained=False, device=self.device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        
        self.class_names = class_names
        return model
    
    def predict_image(self, image: np.ndarray) -> tuple[str, float]:
        if self.use_mediapipe_hands:
            image, _ = self.hand_cropper(image)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = self.class_names[predicted.item()]
        confidence_value = confidence.item()
        
        return predicted_class, confidence_value


def webcam_inference(weights_path: str, img_size: int = 224, use_mediapipe_hands: bool = True, camera: int = 0, confidence_threshold: float = 0.5):
    predictor = ASLPredictor(weights_path=weights_path, img_size=img_size, use_mediapipe_hands=use_mediapipe_hands)
    
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera}")
        return
    
    print("Webcam inference started. Press 'Q' to quit.")
    print(f"Confidence threshold: {confidence_threshold}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            predicted_letter, confidence = predictor.predict_image(frame)
            
            if confidence >= confidence_threshold:
                if use_mediapipe_hands:
                    _, bbox = predictor.hand_cropper(frame)
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                text = f"{predicted_letter} ({confidence:.2f})"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No confident prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("ASL Alphabet Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def image_inference(weights_path: str, image_path: str, img_size: int = 224, use_mediapipe_hands: bool = True):
    predictor = ASLPredictor(weights_path=weights_path, img_size=img_size, use_mediapipe_hands=use_mediapipe_hands)
    
    image_path = Path(image_path)
    
    if image_path.is_file():
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        predicted_letter, confidence = predictor.predict_image(image)
        print(f"Image: {image_path.name}")
        print(f"Prediction: {predicted_letter} (confidence: {confidence:.3f})")
        
    elif image_path.is_dir():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in image_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No image files found in {image_path}")
            return
        
        print(f"Processing {len(image_files)} images...")
        
        for img_file in sorted(image_files):
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"Warning: Could not load {img_file}")
                continue
            
            predicted_letter, confidence = predictor.predict_image(image)
            print(f"{img_file.name}: {predicted_letter} ({confidence:.3f})")
    
    else:
        print(f"Error: {image_path} is not a valid file or directory")


def main():
    parser = argparse.ArgumentParser(description="ASL Alphabet Inference")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--use-mp-hands", action="store_true", help="Use MediaPipe hands for cropping")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for webcam")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--image-path", type=str, help="Path to image or folder for inference")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weights):
        print(f"Error: Weights file {args.weights} not found")
        return
    
    if args.image_path:
        image_inference(
            weights_path=args.weights,
            image_path=args.image_path,
            img_size=args.img_size,
            use_mediapipe_hands=args.use_mp_hands,
        )
    else:
        webcam_inference(
            weights_path=args.weights,
            img_size=args.img_size,
            use_mediapipe_hands=args.use_mp_hands,
            camera=args.camera,
            confidence_threshold=args.conf,
        )


if __name__ == "__main__":
    main() 