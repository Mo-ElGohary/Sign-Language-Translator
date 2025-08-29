from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

try:
	import mediapipe as mp
except Exception:
	mp = None


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def save_checkpoint(
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	scaler: Optional[torch.cuda.amp.GradScaler],
	epoch: int,
	best: bool,
	class_names: List[str],
	checkpoint_dir: str,
	filename: str,
) -> str:
	os.makedirs(checkpoint_dir, exist_ok=True)
	state = {
		"model_state": model.state_dict(),
		"optimizer_state": optimizer.state_dict(),
		"scaler_state": scaler.state_dict() if scaler is not None else None,
		"epoch": epoch,
		"class_names": class_names,
	}
	path = os.path.join(checkpoint_dir, filename)
	torch.save(state, path)
	# Save labels sidecar for convenience
	labels_path = os.path.join(checkpoint_dir, "labels.json")
	with open(labels_path, "w", encoding="utf-8") as f:
		json.dump({"class_names": class_names}, f, indent=2)
	return path


def load_checkpoint(weights_path: str, map_location: Optional[str | torch.device] = None):
	state = torch.load(weights_path, map_location=map_location)
	return state


def compute_metrics(y_true: List[int], y_pred: List[int], class_names: List[str]) -> Tuple[str, np.ndarray]:
	report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
	cm = confusion_matrix(y_true, y_pred)
	return report, cm


class MediaPipeHandCropper:
	"""Optional: Use MediaPipe Hands to crop a tight hand ROI for model input."""

	def __init__(self, static_image_mode: bool = False):
		if mp is None:
			raise RuntimeError("mediapipe is not installed")
		self.hands = mp.solutions.hands.Hands(
			static_image_mode=static_image_mode,
			max_num_hands=1,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5,
		)

	def __call__(self, image_bgr: np.ndarray, fallback_square: bool = True) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
		import cv2

		h, w = image_bgr.shape[:2]
		rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
		res = self.hands.process(rgb)
		if not res.multi_hand_landmarks:
			if fallback_square:
				sz = min(h, w)
				offset_y = (h - sz) // 2
				offset_x = (w - sz) // 2
				crop = image_bgr[offset_y:offset_y+sz, offset_x:offset_x+sz]
				return crop, (offset_x, offset_y, sz, sz)
			return image_bgr, (0, 0, w, h)

		x_min, y_min = w, h
		x_max, y_max = 0, 0
		for lm in res.multi_hand_landmarks:
			for p in lm.landmark:
				x = int(p.x * w)
				y = int(p.y * h)
				x_min = min(x_min, x)
				y_min = min(y_min, y)
				x_max = max(x_max, x)
				y_max = max(y_max, y)
		pad = int(0.2 * max(x_max - x_min, y_max - y_min))
		x0 = max(0, x_min - pad)
		y0 = max(0, y_min - pad)
		x1 = min(w, x_max + pad)
		y1 = min(h, y_max + pad)
		crop = image_bgr[y0:y1, x0:x1]
		return crop, (x0, y0, x1 - x0, y1 - y0) 