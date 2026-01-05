"""
Training loop and utilities.
Includes early stopping, learning rate scheduling, and checkpoint management.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scripts.prepare_data import prepare_ham10000


class EarlyStopper:
    """
    Early stopping handler to prevent overfitting.
    """

    def __init__(self, patience=3, min_delta=1e-4):
        """
        Initialize early stopper.

        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum improvement threshold
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        """
        Check if training should stop.

        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class Trainer:
    """
    Training manager for model training and validation.
    """

    def __init__(self, model, device, criterion=None, checkpoint_dir="./checkpoints"):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            device: torch.device (cuda or cpu)
            criterion: Loss function (default: CrossEntropyLoss)
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.device = device
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def train_epoch(self, train_loader, optimizer):
        """
        Train for one epoch.

        Returns:
            Tuple of (avg_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        with tqdm(train_loader, desc="Training", leave=False) as pbar:
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                pbar.set_postfix({'loss': loss.item():.4f})

        return total_loss / total, correct / total

    def validate(self, val_loader):
        """
        Validate model.

        Returns:
            Tuple of (avg_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

    def train(self, train_loader, val_loader, optimizer, scheduler=None,
              num_epochs=20, patience=3, checkpoint_name="best.pt"):
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler (optional)
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            checkpoint_name: Name of checkpoint file

        Returns:
            Dictionary with training history
        """
        early_stopper = EarlyStopper(patience=patience)
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            val_loss, val_acc = self.validate(val_loader)

            if scheduler:
                scheduler.step()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1:02d} | "
                  f"Train Loss: {train_loss:.4f} / Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f} / Acc: {val_acc:.3f}")

            # Save best model
            if val_loss < early_stopper.best_loss:
                self.save_checkpoint(checkpoint_name)
                print(f"  → Best model saved!")

            # Early stopping
            if early_stopper(val_loss, epoch):
                print(f"Early stopping at epoch {epoch+1}")
                break

        return history

    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        self.model.load_state_dict(torch.load(path, map_location=self.device))
