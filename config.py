from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
	data_dir: str
	checkpoint_dir: str = "checkpoints"
	img_size: int = 224
	batch_size: int = 64
	epochs: int = 10
	learning_rate: float = 3e-4
	weight_decay: float = 0.05
	workers: int = 4
	val_split: float = 0.15
	seed: int = 42
	use_mediapipe_hands: bool = False
	mixed_precision: bool = True
	save_every: int = 1


@dataclass
class InferenceConfig:
	weights: str
	img_size: int = 224
	use_mediapipe_hands: bool = True
	camera: int = 0
	confidence_threshold: float = 0.5
	image_path: str | None = None 