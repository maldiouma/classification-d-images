"""
Grad-CAM visualization for model explainability.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class GradCAMExplainer:
    """
    Grad-CAM explainability tool for visualizing model predictions.
    """
    
    def __init__(self, model, device, layer_name="layer4"):
        """
        Initialize Grad-CAM explainer.
        
        Args:
            model: PyTorch model
            device: torch.device
            layer_name: Name of target layer for attention visualization
        """
        self.model = model
        self.device = device
        self.layer_name = layer_name
        
        # Get target layer
        target_layer = self._get_layer(layer_name)
        self.cam = GradCAM(model=model, target_layers=[target_layer], 
                          use_cuda=(device.type == "cuda"))
    
    def _get_layer(self, layer_name):
        """Get layer by name from model."""
        return dict(self.model.named_modules())[layer_name]
    
    def visualize(self, image_tensor, target_class, class_names=None):
        """
        Visualize Grad-CAM for an image.
        
        Args:
            image_tensor: Image tensor (1, C, H, W)
            target_class: Target class for visualization
            class_names: Optional list of class names
            
        Returns:
            Visualized image with attention map overlay
        """
        # Get Grad-CAM
        grayscale_cam = self.cam(
            input_tensor=image_tensor,
            targets=[ClassifierOutputTarget(target_class)]
        )[0]
        
        # Denormalize image
        image_np = image_tensor[0].cpu().numpy()
        # ImageNet normalization inverse
        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image_np = np.clip(image_np, 0, 1)
        
        # Create visualization
        vis = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
        
        return vis, grayscale_cam
    
    def plot_gradcam(self, image_tensor, target_class, class_names=None, save_path=None):
        """
        Plot Grad-CAM visualization.
        
        Args:
            image_tensor: Image tensor (1, C, H, W)
            target_class: Target class index
            class_names: Optional class names
            save_path: Path to save figure
        """
        vis, grayscale_cam = self.visualize(image_tensor, target_class, class_names)
        
        class_name = class_names[target_class] if class_names else f"Class {target_class}"
        
        plt.figure(figsize=(12, 4))
        
        # Original image
        image_np = image_tensor[0].cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image_np = np.clip(image_np, 0, 1)
        
        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        plt.title('Original Image')
        plt.axis('off')
        
        # Attention map
        plt.subplot(1, 3, 2)
        plt.imshow(grayscale_cam, cmap='hot')
        plt.title('Attention Map')
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 3)
        plt.imshow(vis)
        plt.title(f'Prediction: {class_name}')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def batch_visualize(self, image_batch, predictions, class_names=None, save_path=None):
        """
        Visualize Grad-CAM for a batch of images.
        
        Args:
            image_batch: Batch of images (N, C, H, W)
            predictions: Model predictions (N,)
            class_names: Optional class names
            save_path: Path pattern to save figures (e.g., 'output_{:03d}.png')
        """
        batch_size = image_batch.size(0)
        cols = min(batch_size, 3)
        rows = (batch_size + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols * 2, figsize=(12, 4 * rows))
        
        for idx in range(batch_size):
            image = image_batch[idx:idx+1]
            pred_class = predictions[idx].item()
            
            vis, cam = self.visualize(image, pred_class, class_names)
            
            # Original
            row = idx // cols
            col = (idx % cols) * 2
            
            image_np = image[0].cpu().numpy()
            image_np = np.transpose(image_np, (1, 2, 0))
            image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image_np = np.clip(image_np, 0, 1)
            
            axes[row, col].imshow(image_np)
            axes[row, col].set_title(f'Image {idx}')
            axes[row, col].axis('off')
            
            # Grad-CAM
            axes[row, col+1].imshow(vis)
            class_name = class_names[pred_class] if class_names else f"Class {pred_class}"
            axes[row, col+1].set_title(f'Pred: {class_name}')
            axes[row, col+1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
