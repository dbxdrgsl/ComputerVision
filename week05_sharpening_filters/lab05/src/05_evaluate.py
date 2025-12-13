import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from dataset import load_dataset

import matplotlib.pyplot as plt

# Fix path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate U-Net on test set")
    parser.add_argument("--model_path", type=str, default="models/unet_best.keras")
    parser.add_argument("--split_file", type=str, default="runs/split_test.txt")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="runs/eval")
    parser.add_argument("--max_vis", type=int, default=10)
    return parser.parse_args()

def compute_metrics(y_true, y_pred):
    """Compute TP, TN, FP, FN for binary segmentation"""
    y_true = y_true.flatten().astype(np.int32)
    y_pred = y_pred.flatten().astype(np.int32)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    return tp, tn, fp, fn

def pixel_accuracy(tp, tn, fp, fn):
    """PA = (TP + TN) / (TP + TN + FP + FN)"""
    denom = tp + tn + fp + fn
    if denom == 0:
        return 0.0
    return (tp + tn) / denom

def iou_jaccard(tp, tn, fp, fn):
    """IoU = TP / (TP + FP + FN)"""
    denom = tp + fp + fn
    if denom == 0:
        return 0.0
    return tp / denom

def dice_coefficient(tp, tn, fp, fn):
    """Dice = 2*TP / (2*TP + FP + FN)"""
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return (2 * tp) / denom

def load_test_images(split_file, img_size):
    """Load test image paths from split file"""
    test_paths = []
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            test_paths = [line.strip() for line in f.readlines()]
    return test_paths

def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path, compile=False)
    
    # Load test dataset
    print(f"Loading test dataset from {args.split_file}...")
    test_dataset = load_dataset(args.split_file, args.img_size, batch_size=1, shuffle=False)
    
    # Count test samples
    test_samples = len(load_test_images(args.split_file, args.img_size))
    print(f"Test samples: {test_samples}\n")
    
    # Accumulate metrics
    all_pa = []
    all_iou = []
    all_dice = []
    
    # Store samples for visualization
    vis_images = []
    vis_gts = []
    vis_preds = []
    
    # Evaluation loop
    for idx, (images, masks) in enumerate(test_dataset):
        # Forward pass
        predictions = model(images, training=False)
        
        # Threshold at 0.5
        pred_binary = (predictions.numpy() > 0.5).astype(np.uint8)
        
        # Get ground truth
        gt_binary = masks.numpy().astype(np.uint8)
        
        # Compute metrics
        tp, tn, fp, fn = compute_metrics(gt_binary, pred_binary)
        pa = pixel_accuracy(tp, tn, fp, fn)
        iou = iou_jaccard(tp, tn, fp, fn)
        dice = dice_coefficient(tp, tn, fp, fn)
        
        all_pa.append(pa)
        all_iou.append(iou)
        all_dice.append(dice)
        
        # Store for visualization
        if len(vis_images) < args.max_vis:
            vis_images.append(images.numpy()[0])
            vis_gts.append(gt_binary[0])
            vis_preds.append(pred_binary[0])
    
    # Compute means
    mean_pa = np.mean(all_pa)
    mean_iou = np.mean(all_iou)
    mean_dice = np.mean(all_dice)
    
    # Print results
    print("=" * 50)
    print(f"Mean Pixel Accuracy: {mean_pa:.4f}")
    print(f"Mean IoU (Jaccard):  {mean_iou:.4f}")
    print(f"Mean Dice:           {mean_dice:.4f}")
    print("=" * 50)
    
    # Qualitative visualization
    num_vis = len(vis_images)
    fig, axes = plt.subplots(num_vis, 4, figsize=(16, 4 * num_vis))
    
    if num_vis == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_vis):
        # Original image
        axes[i, 0].imshow(vis_images[i])
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")
        
        # Ground truth
        axes[i, 1].imshow(vis_gts[i], cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")
        
        # Prediction
        axes[i, 2].imshow(vis_preds[i], cmap="gray")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")
        
        # Overlay
        overlay = vis_images[i].copy()
        mask_overlay = vis_preds[i].astype(float)
        overlay[..., 0] = np.clip(overlay[..., 0] + mask_overlay * 0.4, 0, 1)
        overlay[..., 1] = np.clip(overlay[..., 1] - mask_overlay * 0.2, 0, 1)
        overlay[..., 2] = np.clip(overlay[..., 2] - mask_overlay * 0.2, 0, 1)
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("Overlay (Pred)")
        axes[i, 3].axis("off")
    
    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "qualitative_results.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"\nQualitative results saved to {out_path}")

if __name__ == "__main__":
    main()