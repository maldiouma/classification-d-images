#!/usr/bin/env python3
"""
Inference script for medical image classification.
Performs prediction, visualization, Grad-CAM analysis, and error analysis.

Usage:
    python scripts/inference.py --image path/to/image.jpg --checkpoint best_ft.pt
    python scripts/inference.py --test-dir ./data/test --checkpoint best_ft.pt
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import build_model
from evaluation import Evaluator
from gradcam import GradCAMExplainer
from data import MedicalImageDataset
import torchvision.transforms as transforms


def load_image(image_path, img_size=224):
    """Load and preprocess a single image."""
    image = Image.open(image_path).convert('RGB')
    
    tfm = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return tfm(image).unsqueeze(0)


def predict_single_image(model, image_tensor, device, class_names):
    """Predict class for a single image."""
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = outputs.argmax(1).item()
        pred_prob = probs[0, pred_class].item()
    
    class_name = class_names[pred_class]
    
    return pred_class, pred_prob, probs[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Inference for medical image classification")
    
    parser.add_argument('--image', type=str, 
                       help='Path to single image for inference')
    parser.add_argument('--test-dir', type=str, 
                       help='Path to test directory for batch inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='./outputs/config.json',
                       help='Path to training config')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--gradcam', action='store_true',
                       help='Generate Grad-CAM visualization')
    parser.add_argument('--output-dir', type=str, default='./inference_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found at {config_path}")
        print("Using default config...")
        class_names = ['Class_0', 'Class_1']
        num_classes = 2
    else:
        with open(config_path) as f:
            config = json.load(f)
        class_names = config.get('class_names', ['Class_0', 'Class_1'])
        num_classes = config.get('num_classes', len(class_names))
    
    print(f"Classes: {class_names}")
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        # Try to find in checkpoints directory
        checkpoint_path = Path('./outputs/checkpoints') / args.checkpoint
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    
    model = build_model(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Model loaded from {checkpoint_path}")
    
    # Setup Grad-CAM if requested
    if args.gradcam:
        explainer = GradCAMExplainer(model, device)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Single image inference
    if args.image:
        print(f"\nProcessing image: {args.image}")
        
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return
        
        image_tensor = load_image(str(image_path), args.img_size)
        pred_class, pred_prob, all_probs = predict_single_image(model, image_tensor, device, class_names)
        
        print(f"\nPrediction:")
        print(f"  Class: {class_names[pred_class]}")
        print(f"  Confidence: {pred_prob:.4f}")
        print(f"\nAll probabilities:")
        for cls_name, prob in zip(class_names, all_probs):
            print(f"  {cls_name}: {prob:.4f}")
        
        # Grad-CAM visualization
        if args.gradcam:
            save_path = output_dir / f"gradcam_{image_path.stem}.png"
            explainer.plot_gradcam(image_tensor.to(device), pred_class, class_names, save_path)
    
    # Batch inference on test directory
    if args.test_dir:
        print(f"\nBatch inference on: {args.test_dir}")
        
        dataset = MedicalImageDataset(data_root=args.test_dir, img_size=args.img_size)
        try:
            _, _, test_loader, class_names, num_classes = dataset.load_dataloaders()
        except Exception as e:
            print(f"Error loading test data: {e}")
            return
        
        evaluator = Evaluator(model, device, num_classes=num_classes)
        y_true, y_pred, y_proba = evaluator.get_predictions(test_loader)
        metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
        
        print("\nMetrics on test set:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        # Save visualizations
        evaluator.plot_confusion_matrix(
            y_true, y_pred, class_names,
            save_path=output_dir / "confusion_matrix.png"
        )
        
        if num_classes == 2:
            evaluator.plot_roc_curve(y_true, y_proba, save_path=output_dir / "roc_curve.png")
            evaluator.plot_pr_curve(y_true, y_proba, save_path=output_dir / "pr_curve.png")
            evaluator.calibration_curve(y_true, y_proba, save_path=output_dir / "calibration_curve.png")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
