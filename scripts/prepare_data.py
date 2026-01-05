#!/usr/bin/env python3
"""
Dataset preparation script for HAM10000 skin lesions dataset.
Organizes raw dataset into train/val/test splits with stratification.
"""

import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def prepare_ham10000(raw_data_path="./data/HAM10000_images", output_path="./data"):
    """
    Prepare HAM10000 dataset with stratified train/val/test split.
    
    Args:
        raw_data_path: Path to raw HAM10000 images directory
        output_path: Output directory for organized splits
    """
    
    raw_path = Path(raw_data_path)
    output_dir = Path(output_path)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = list(raw_path.glob('*.jpg'))
    
    if not image_files:
        print(f"No images found in {raw_path}")
        return
    
    # Extract class from filename (HAM10000 format: IMGID_CLASS.jpg)
    classes = {}
    for img_path in image_files:
        stem = img_path.stem
        # Extract class label (format: IMGID_dx)
        class_label = stem.split('_')[1] if '_' in stem else 'unknown'
        
        if class_label not in classes:
            classes[class_label] = []
        classes[class_label].append(img_path)
    
    print(f"\nDataset statistics:")
    print(f"Total images: {len(image_files)}")
    print(f"Number of classes: {len(classes)}")
    print("\nClass distribution:")
    
    all_images = []
    all_labels = []
    
    for class_label, images in sorted(classes.items()):
        print(f"  {class_label}: {len(images)} images")
        all_images.extend(images)
        all_labels.extend([class_label] * len(images))
    
    # Stratified split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_images, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nStratified split:")
    print(f"  Train: {len(X_train)} images ({100*len(X_train)/len(all_images):.1f}%)")
    print(f"  Val:   {len(X_val)} images ({100*len(X_val)/len(all_images):.1f}%)")
    print(f"  Test:  {len(X_test)} images ({100*len(X_test)/len(all_images):.1f}%)")
    
    # Copy files to splits
    splits = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
    
    for split_name, (images, labels) in splits.items():
        print(f"\nOrganizing {split_name} set...")
        
        # Create class subdirectories
        for class_label in classes.keys():
            (output_dir / split_name / class_label).mkdir(parents=True, exist_ok=True)
        
        # Copy images
        for img_path, class_label in tqdm(zip(images, labels), total=len(images), leave=False):
            dest_path = output_dir / split_name / class_label / img_path.name
            shutil.copy2(img_path, dest_path)
    
    print(f"\n✓ Dataset successfully prepared in {output_dir}")
    print(f"  Structure:")
    print(f"    {output_dir}/train/")
    print(f"    {output_dir}/val/")
    print(f"    {output_dir}/test/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare HAM10000 dataset")
    parser.add_argument('--raw-path', type=str, default='./data/HAM10000_images',
                       help='Path to raw HAM10000 images')
    parser.add_argument('--output-path', type=str, default='./data',
                       help='Output path for organized dataset')
    
    args = parser.parse_args()
    
    prepare_ham10000(args.raw_path, args.output_path)
