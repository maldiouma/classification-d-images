"""
Model definition and transfer learning utilities.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes=2, model_name="resnet18", freeze_backbone=True, dropout_rate=0.2):
    """
    Build a transfer learning model with ResNet backbone.
    
    Args:
        num_classes: Number of output classes
        model_name: Model architecture ('resnet18' or 'resnet50')
        freeze_backbone: Whether to freeze backbone weights
        dropout_rate: Dropout rate in classification head
        
    Returns:
        PyTorch model
    """
    
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace classification head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )
    
    return model


def unfreeze_layer4(model):
    """
    Unfreeze the last residual block (layer4) for fine-tuning.
    
    Args:
        model: PyTorch model
    """
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True


def count_parameters(model):
    """
    Count trainable and total parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (trainable_params, total_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
