from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ASLClassifier(nn.Module):
	"""ASL alphabet classifier using MobileNetV3 as backbone."""
	
	def __init__(self, num_classes: int, pretrained: bool = True):
		super().__init__()
		
		# Load pretrained MobileNetV3
		self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
		
		# Replace the classifier head
		in_features = self.backbone.classifier[-1].in_features
		self.backbone.classifier = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(in_features, 512),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.2),
			nn.Linear(512, num_classes)
		)
		
		# Initialize the new classifier layers
		self._init_classifier()
	
	def _init_classifier(self):
		"""Initialize the classifier layers with proper weights."""
		for m in self.backbone.classifier.modules():
			if isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.backbone(x)
	
	def get_features(self, x: torch.Tensor) -> torch.Tensor:
		"""Extract features before the classifier head."""
		# Remove the classifier and get features
		features = self.backbone.features(x)
		features = self.backbone.avgpool(features)
		features = torch.flatten(features, 1)
		return features


def create_model(
	num_classes: int,
	pretrained: bool = True,
	device: torch.device | None = None,
) -> ASLClassifier:
	"""Create and return an ASL classifier model."""
	
	model = ASLClassifier(num_classes=num_classes, pretrained=pretrained)
	
	if device is not None:
		model = model.to(device)
	
	return model


def count_parameters(model: nn.Module) -> int:
	"""Count the number of trainable parameters in the model."""
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_backbone(model: ASLClassifier, freeze: bool = True):
	"""Freeze or unfreeze the backbone layers."""
	for param in model.backbone.features.parameters():
		param.requires_grad = not freeze
	
	if freeze:
		print("Backbone layers frozen")
	else:
		print("Backbone layers unfrozen")


def get_model_summary(model: ASLClassifier) -> str:
	"""Get a summary of the model architecture."""
	summary = []
	summary.append(f"Model: ASLClassifier")
	summary.append(f"Backbone: MobileNetV3-Small")
	summary.append(f"Total parameters: {count_parameters(model):,}")
	summary.append(f"Trainable parameters: {count_parameters(model):,}")
	
	# Count backbone parameters
	backbone_params = sum(p.numel() for p in model.backbone.features.parameters())
	summary.append(f"Backbone parameters: {backbone_params:,}")
	
	# Count classifier parameters
	classifier_params = sum(p.numel() for p in model.backbone.classifier.parameters())
	summary.append(f"Classifier parameters: {classifier_params:,}")
	
	return "\n".join(summary) 