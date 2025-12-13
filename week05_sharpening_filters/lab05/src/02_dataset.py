import argparse
import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
from PIL import Image

import torch.utils.data
import matplotlib.pyplot as plt


class SegmentationDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for medical image segmentation."""
    
    def __init__(
        self,
        split_file: str,
        img_size: Tuple[int, int] = (256, 256),
        num_classes: int = 1,
        augment: bool = False
    ):
        """
        Args:
            split_file: Path to split txt file (image_path mask_path per line)
            img_size: Target image size (H, W)
            num_classes: Number of classes (1 for binary, >1 for multi-class)
            augment: Apply augmentation if True
        """
        self.split_file = split_file
        self.img_size = img_size
        self.num_classes = num_classes
        self.augment = augment
        self.repo_root = Path(__file__).parent.parent
        
        # Load and validate file pairs
        self.file_pairs = self._load_split_file()
        
    def _load_split_file(self) -> list:
        """Load and validate split file."""
        pairs = []
        split_path = self.repo_root / self.split_file
        
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        
        with open(split_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid line format: {line}")
                img_path, mask_path = parts
                
                img_full = self.repo_root / img_path
                mask_full = self.repo_root / mask_path
                
                if not img_full.exists():
                    raise FileNotFoundError(f"Image not found: {img_full}")
                if not mask_full.exists():
                    raise FileNotFoundError(f"Mask not found: {mask_full}")
                
                pairs.append((img_full, mask_full))
        
        if not pairs:
            raise ValueError(f"No file pairs found in {split_path}")
        
        return pairs
    
    def _load_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess image."""
        img = Image.open(path).convert('RGB')
        img = img.resize(self.img_size, Image.BILINEAR)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (3, H, W)
        return img_tensor
    
    def _load_mask(self, path: Path) -> np.ndarray:
        """Load mask as grayscale numpy array."""
        mask = Image.open(path).convert('L')
        mask = mask.resize(self.img_size, Image.NEAREST)
        mask_array = np.array(mask, dtype=np.uint8)
        return mask_array
    
    def _normalize_mask(self, mask_array: np.ndarray) -> np.ndarray:
        """Normalize mask to {0, 1}."""
        unique_vals = np.unique(mask_array)
        
        # Handle {0, 255} case
        if len(unique_vals) == 2 and 255 in unique_vals:
            mask_array = (mask_array > 127).astype(np.uint8)
        # Handle arbitrary grayscale: threshold at 0
        else:
            mask_array = (mask_array > 0).astype(np.uint8)
        
        # Validate binary
        assert set(np.unique(mask_array)).issubset({0, 1}), \
            f"Mask not binary after normalization: {np.unique(mask_array)}"
        
        return mask_array
    
    def _apply_augmentation(
        self,
        image: torch.Tensor,
        mask: np.ndarray
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Apply augmentation (horizontal/vertical flip)."""
        if np.random.rand() > 0.5:  # Horizontal flip
            image = torch.flip(image, dims=[2])
            mask = np.fliplr(mask)
        
        if np.random.rand() > 0.5:  # Vertical flip
            image = torch.flip(image, dims=[1])
            mask = np.flipud(mask)
        
        return image, mask
    
    def __len__(self) -> int:
        return len(self.file_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (image_tensor, mask_tensor)."""
        img_path, mask_path = self.file_pairs[idx]
        
        # Load image
        image = self._load_image(img_path)
        
        # Load and normalize mask
        mask_array = self._load_mask(mask_path)
        mask_array = self._normalize_mask(mask_array)
        
        # Apply augmentation
        if self.augment:
            image, mask_array = self._apply_augmentation(image, mask_array)
        
        # Convert mask to tensor
        if self.num_classes == 1:
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).float()  # (1, H, W)
        else:
            # One-hot encoding
            mask_tensor = torch.zeros(self.num_classes, *self.img_size, dtype=torch.long)
            for c in range(self.num_classes):
                mask_tensor[c] = torch.from_numpy((mask_array == c).astype(np.uint8))
        
        return image, mask_tensor


def visualize_batch(image: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Create visualization with image, mask, and overlay."""
    # Convert to numpy
    img_np = image.permute(1, 2, 0).numpy()  # (H, W, 3)
    mask_np = mask.squeeze().numpy()  # (H, W)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Image
    axes[0].imshow(img_np)
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    # Mask
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')
    
    # Overlay
    overlay = img_np.copy()
    red_mask = np.zeros_like(img_np)
    red_mask[..., 0] = mask_np  # Red channel
    overlay = (1 - 0.4) * overlay + 0.4 * red_mask
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (Red)')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Medical Image Segmentation Dataset')
    parser.add_argument('--split_file', default='runs/split_train.txt', help='Path to split file')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--no_vis', action='store_true', help='Skip visualization')
    args = parser.parse_args()
    
    # Create dataset
    print(f"Loading dataset from {args.split_file}...")
    dataset = SegmentationDataset(
        split_file=args.split_file,
        img_size=(args.img_size, args.img_size),
        num_classes=args.num_classes,
        augment=False
    )
    print(f"✓ Loaded {len(dataset)} samples\n")
    
    # Create dataloader
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Get one batch
    image, mask = next(iter(loader))
    
    # Print summary
    print("=" * 50)
    print("DATASET CHECKPOINT")
    print("=" * 50)
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}, range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask dtype: {mask.dtype}")
    print(f"Unique mask values: {torch.unique(mask).tolist()}")
    print("=" * 50 + "\n")
    
    # Visualize
    if not args.no_vis:
        try:
            fig = visualize_batch(image[0], mask[0])
            save_path = Path(__file__).parent.parent / 'runs' / 'dataset_check.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
            plt.close(fig)
        except Exception as e:
            print(f"✗ Visualization failed: {e}")


if __name__ == '__main__':
    main()