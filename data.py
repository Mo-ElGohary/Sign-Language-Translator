from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .utils import MediaPipeHandCropper


class ASLDataset(Dataset):
	"""Custom dataset that optionally applies MediaPipe hand cropping."""

	def __init__(
		self,
		root: str,
		transform: Optional[transforms.Compose] = None,
		use_mediapipe_hands: bool = False,
	):
		self.dataset = ImageFolder(root=root)
		self.transform = transform
		self.use_mediapipe_hands = use_mediapipe_hands
		if use_mediapipe_hands:
			self.hand_cropper = MediaPipeHandCropper(static_image_mode=True)

	def __len__(self) -> int:
		return len(self.dataset)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
		img, label = self.dataset[idx]
		
		if self.use_mediapipe_hands:
			# Convert PIL to OpenCV format
			img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
			# Apply hand cropping
			img_cropped, _ = self.hand_cropper(img_cv)
			# Convert back to PIL
			img_cropped_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
			img = transforms.ToPILImage()(img_cropped_rgb)
		
		if self.transform:
			img = self.transform(img)
		
		return img, label

	@property
	def classes(self) -> List[str]:
		return self.dataset.classes


def get_transforms(
	img_size: int = 224,
	is_training: bool = True,
) -> transforms.Compose:
	"""Get image transforms for training or validation."""
	
	if is_training:
		transform = transforms.Compose([
			transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
			transforms.RandomRotation(degrees=15),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
	else:
		transform = transforms.Compose([
			transforms.Resize((img_size, img_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
	
	return transform


def split_dataset(
	data_dir: str,
	val_split: float = 0.15,
	seed: int = 42,
) -> Tuple[str, str]:
	"""Split a single dataset directory into train/val directories."""
	
	data_path = Path(data_dir)
	if not data_path.exists():
		raise ValueError(f"Data directory {data_dir} does not exist")
	
	# Check if already split
	train_dir = data_path / "train"
	val_dir = data_path / "val"
	
	if train_dir.exists() and val_dir.exists():
		return str(train_dir), str(val_dir)
	
	# Create split directories
	train_dir.mkdir(exist_ok=True)
	val_dir.mkdir(exist_ok=True)
	
	# Get all class directories
	class_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name not in ["train", "val"]]
	
	if not class_dirs:
		raise ValueError(f"No class directories found in {data_dir}")
	
	random.seed(seed)
	
	for class_dir in class_dirs:
		class_name = class_dir.name
		
		# Create class directories in train and val
		(train_dir / class_name).mkdir(exist_ok=True)
		(val_dir / class_name).mkdir(exist_ok=True)
		
		# Get all images in class directory
		image_files = list(class_dir.glob("*"))
		image_files = [f for f in image_files if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
		
		if not image_files:
			print(f"Warning: No images found in {class_dir}")
			continue
		
		# Shuffle and split
		random.shuffle(image_files)
		split_idx = int(len(image_files) * (1 - val_split))
		
		train_files = image_files[:split_idx]
		val_files = image_files[split_idx:]
		
		# Copy files
		for file in train_files:
			shutil.copy2(file, train_dir / class_name / file.name)
		
		for file in val_files:
			shutil.copy2(file, val_dir / class_name / file.name)
		
		print(f"Class {class_name}: {len(train_files)} train, {len(val_files)} val")
	
	return str(train_dir), str(val_dir)


def create_data_loaders(
	data_dir: str,
	img_size: int = 224,
	batch_size: int = 64,
	workers: int = 4,
	val_split: float = 0.15,
	use_mediapipe_hands: bool = False,
	seed: int = 42,
) -> Tuple[DataLoader, DataLoader, List[str]]:
	"""Create train and validation data loaders."""
	
	# Split dataset if needed
	train_dir, val_dir = split_dataset(data_dir, val_split, seed)
	
	# Get transforms
	train_transform = get_transforms(img_size, is_training=True)
	val_transform = get_transforms(img_size, is_training=False)
	
	# Create datasets
	train_dataset = ASLDataset(
		root=train_dir,
		transform=train_transform,
		use_mediapipe_hands=use_mediapipe_hands,
	)
	
	val_dataset = ASLDataset(
		root=val_dir,
		transform=val_transform,
		use_mediapipe_hands=use_mediapipe_hands,
	)
	
	# Create data loaders
	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=workers,
		pin_memory=True,
		drop_last=True,
	)
	
	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=workers,
		pin_memory=True,
		drop_last=False,
	)
	
	print(f"Train samples: {len(train_dataset)}")
	print(f"Val samples: {len(val_dataset)}")
	print(f"Classes: {train_dataset.classes}")
	
	return train_loader, val_loader, train_dataset.classes 