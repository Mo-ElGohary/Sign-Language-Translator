#!/usr/bin/env python3
"""
ASL Alphabet Training Script
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from config import TrainConfig
from data import create_data_loaders
from model import create_model, get_model_summary
from utils import compute_metrics, get_device, save_checkpoint, set_seed


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        acc = 100.0 * correct / total
        pbar.set_postfix({'Loss': f'{total_loss/(batch_idx+1):.4f}', 'Acc': f'{acc:.2f}%'})
    
    return total_loss / len(train_loader), 100.0 * correct / total


def validate_epoch(model, val_loader, criterion, device, epoch, class_names):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            acc = 100.0 * correct / total
            pbar.set_postfix({'Loss': f'{total_loss/(batch_idx+1):.4f}', 'Acc': f'{acc:.2f}%'})
    
    report, _ = compute_metrics(all_targets, all_predictions, class_names)
    return total_loss / len(val_loader), 100.0 * correct / total, report


def main():
    parser = argparse.ArgumentParser(description="Train ASL Alphabet Classifier")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-mp-hands", action="store_true", help="Use MediaPipe hands for cropping")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    
    print("Creating data loaders...")
    train_loader, val_loader, class_names = create_data_loaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        workers=args.workers,
        val_split=args.val_split,
        use_mediapipe_hands=args.use_mp_hands,
        seed=args.seed,
    )
    
    print("Creating model...")
    model = create_model(num_classes=len(class_names), pretrained=True, device=device)
    print(get_model_summary(model))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler() if not args.no_mixed_precision else None
    
    best_acc = 0.0
    best_epoch = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Mixed precision: {not args.no_mixed_precision}")
    print(f"MediaPipe hands: {args.use_mp_hands}")
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_acc, report = validate_epoch(model, val_loader, criterion, device, epoch, class_names)
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            save_checkpoint(model, optimizer, scaler, epoch, True, class_names, args.checkpoint_dir, "best.pt")
            print(f"  New best model saved! (Acc: {best_acc:.2f}%)")
        
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, scaler, epoch, False, class_names, args.checkpoint_dir, "last.pt")
        
        if (epoch + 1) % 5 == 0:
            print("\nDetailed Validation Report:")
            print(report)
    
    save_checkpoint(model, optimizer, scaler, args.epochs - 1, False, class_names, args.checkpoint_dir, "final.pt")
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.2f}% (epoch {best_epoch+1})")
    print(f"Models saved in: {args.checkpoint_dir}")


if __name__ == "__main__":
    main() 