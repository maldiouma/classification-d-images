#!/usr/bin/env python3
"""
Training script for medical image classification.
Handles data loading, model training, fine-tuning, and validation.

Usage:
    python scripts/train.py --config config.json
    python scripts/train.py --model resnet50 --epochs 20 --batch-size 64
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import json
from datetime import datetime

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import MedicalImageDataset
from models import build_model, unfreeze_layer4, count_parameters
from training import Trainer
from evaluation import Evaluator


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    parser = argparse.ArgumentParser(description="Train medical image classification model")
    
    # Data arguments
    parser.add_argument('--data-root', type=str, default='./data', 
                       help='Root directory for dataset')
    parser.add_argument('--img-size', type=int, default=224, 
                       help='Image size for resizing')
    parser.add_argument('--batch-size', type=int, default=64, 
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=2, 
                       help='Number of workers for DataLoader')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet18', 
                       choices=['resnet18', 'resnet50'],
                       help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=7, 
                       help='Number of output classes')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20, 
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, 
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, 
                       help='Weight decay for optimizer')
    parser.add_argument('--dropout', type=float, default=0.2, 
                       help='Dropout rate')
    parser.add_argument('--patience', type=int, default=3, 
                       help='Early stopping patience')
    
    # Fine-tuning arguments
    parser.add_argument('--fine-tune-epochs', type=int, default=10, 
                       help='Number of fine-tuning epochs')
    parser.add_argument('--ft-lr', type=float, default=5e-4, 
                       help='Learning rate for fine-tuning')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='./outputs', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n" + "="*60)
    print("Loading data...")
    print("="*60)
    
    dataset = MedicalImageDataset(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    try:
        train_loader, val_loader, test_loader, class_names, num_classes = dataset.load_dataloaders()
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Make sure dataset is in {args.data_root}/train, {args.data_root}/val, {args.data_root}/test")
        return
    
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Calculate class weights
    class_weights = dataset.get_class_weights(train_loader, num_classes)
    print(f"Class weights: {class_weights}")
    
    # Build model
    print("\n" + "="*60)
    print("Building model...")
    print("="*60)
    
    model = build_model(
        num_classes=num_classes,
        model_name=args.model,
        freeze_backbone=True,
        dropout_rate=args.dropout
    ).to(device)
    
    trainable, total = count_parameters(model)
    print(f"Model: {args.model}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters: {total:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    trainer = Trainer(model, device, criterion=criterion, checkpoint_dir=str(checkpoint_dir))
    
    # Stage 1: Train classification head
    print("\n" + "="*60)
    print("Stage 1: Training classification head...")
    print("="*60)
    
    history = trainer.train(
        train_loader, val_loader, optimizer, scheduler,
        num_epochs=args.epochs, patience=args.patience,
        checkpoint_name="best.pt"
    )
    
    # Stage 2: Fine-tuning
    print("\n" + "="*60)
    print("Stage 2: Fine-tuning (unfreezing layer4)...")
    print("="*60)
    
    trainer.load_checkpoint("best.pt")
    unfreeze_layer4(model)
    
    # Count parameters again
    trainable, total = count_parameters(model)
    print(f"Trainable parameters after unfreezing: {trainable:,}")
    
    # Create new optimizer for fine-tuning
    ft_optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.ft_lr,
        weight_decay=args.weight_decay
    )
    ft_scheduler = CosineAnnealingLR(ft_optimizer, T_max=args.fine_tune_epochs)
    
    ft_history = trainer.train(
        train_loader, val_loader, ft_optimizer, ft_scheduler,
        num_epochs=args.fine_tune_epochs, patience=2,
        checkpoint_name="best_ft.pt"
    )
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("Final evaluation on test set...")
    print("="*60)
    
    trainer.load_checkpoint("best_ft.pt")
    evaluator = Evaluator(model, device, num_classes=num_classes)
    
    y_true, y_pred, y_proba = evaluator.get_predictions(test_loader)
    metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
    
    print("\nTest Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Save metrics and config
    config = vars(args)
    config['metrics'] = metrics
    config['class_names'] = class_names
    
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
