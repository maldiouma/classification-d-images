"""
Data loading and preprocessing module.
Handles dataset loading, augmentation, and DataLoader creation.
"""

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np


class MedicalImageDataset:
    """
    Medical image dataset handler with augmentation and stratified split.
    """
    
    def __init__(self, data_root="./data", img_size=224, batch_size=64, num_workers=2):
        """
        Initialize dataset handler.
        
        Args:
            data_root: Root directory containing train/val/test subdirectories
            img_size: Image size for resizing
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
        """
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define transformations
        self.train_tfms = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.valid_tfms = transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_dataloaders(self, dataset_type="ImageFolder"):
        """
        Load data from ImageFolder structure (train/val/test subdirectories).
        
        Args:
            dataset_type: Type of dataset structure ('ImageFolder')
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader, class_names, num_classes)
        """
        
        train_path = self.data_root / "train"
        val_path = self.data_root / "val"
        test_path = self.data_root / "test"
        
        # Load datasets
        train_ds = datasets.ImageFolder(str(train_path), transform=self.train_tfms)
        val_ds = datasets.ImageFolder(str(val_path), transform=self.valid_tfms)
        test_ds = datasets.ImageFolder(str(test_path), transform=self.valid_tfms)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        class_names = train_ds.classes
        num_classes = len(class_names)
        
        return train_loader, val_loader, test_loader, class_names, num_classes
    
    def get_class_weights(self, train_loader, num_classes):
        """
        Calculate class weights for handling imbalanced datasets.
        
        Args:
            train_loader: Training DataLoader
            num_classes: Number of classes
            
        Returns:
            Tensor of class weights
        """
        class_counts = torch.zeros(num_classes)
        
        for _, labels in train_loader:
            for label in labels:
                class_counts[label] += 1
        
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * num_classes
        
        return class_weights
