"""
Model evaluation and metrics calculation.
Includes ROC-AUC, PR-AUC, F1-score, calibration, and confusion matrix.
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, accuracy_score, precision_score, recall_score,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator:
    """
    Model evaluation utilities.
    """
    
    def __init__(self, model, device, num_classes=2):
        """
        Initialize evaluator.
        
        Args:
            model: PyTorch model
            device: torch.device
            num_classes: Number of classes
        """
        self.model = model
        self.device = device
        self.num_classes = num_classes
    
    def get_predictions(self, data_loader):
        """
        Get predictions and probabilities on a dataset.
        
        Args:
            data_loader: PyTorch DataLoader
            
        Returns:
            Tuple of (y_true, y_pred, y_proba)
        """
        self.model.eval()
        y_true = []
        y_proba = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                
                y_true.extend(labels.numpy().tolist())
                if self.num_classes == 2:
                    y_proba.extend(probs[:, 1].cpu().numpy().tolist())
                else:
                    y_proba.extend(probs.cpu().numpy().tolist())
        
        y_true = np.array(y_true)
        y_proba = np.array(y_proba)
        
        # Get predictions
        if self.num_classes == 2:
            y_pred = (y_proba >= 0.5).astype(int)
        else:
            y_pred = np.argmax(y_proba, axis=1)
        
        return y_true, y_pred, y_proba
    
    def compute_metrics(self, y_true, y_pred, y_proba):
        """
        Compute evaluation metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        if self.num_classes == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        else:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            metrics['pr_auc'] = average_precision_score(y_true, y_proba, average='weighted')
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, save_path=None):
        """
        Plot confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, y_true, y_proba, save_path=None):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pr_curve(self, y_true, y_proba, save_path=None):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def calibration_curve(self, y_true, y_proba, n_bins=10, save_path=None):
        """Plot calibration curve."""
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, 's-', label='ResNet18')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
