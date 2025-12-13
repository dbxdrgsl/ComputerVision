import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Set

SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
MASK_SUFFIXES = {'_mask', '-mask', 'mask', '_seg', '-seg'}
MASK_PREFIXES = {'mask_'}


def find_files(directory: Path, extensions: Set[str]) -> List[Path]:
    """Recursively find all files with supported extensions."""
    files = []
    for ext in extensions:
        files.extend(directory.rglob(f'*{ext}'))
        files.extend(directory.rglob(f'*{ext.upper()}'))
    return sorted(set(files))


def index_by_stem(files: List[Path]) -> Dict[str, Path]:
    """Index files by stem (filename without extension)."""
    index = {}
    for f in files:
        stem = f.stem
        index[stem] = f
    return index


def get_mask_candidates(stem: str, mask_index: Dict[str, Path]) -> List[Tuple[str, int]]:
    """Get candidate mask stems and their match priority (lower is better)."""
    candidates = []
    
    # Exact match
    if stem in mask_index:
        candidates.append((stem, 0))
    
    # Try removing suffixes
    for suffix in MASK_SUFFIXES:
        if stem.endswith(suffix):
            base = stem[:-len(suffix)]
            if base in mask_index:
                candidates.append((base, 1))
    
    # Try removing prefixes
    for prefix in MASK_PREFIXES:
        if stem.startswith(prefix):
            base = stem[len(prefix):]
            if base in mask_index:
                candidates.append((base, 1))
    
    # Try adding suffixes
    for suffix in MASK_SUFFIXES:
        candidate = stem + suffix
        if candidate in mask_index:
            candidates.append((candidate, 2))
    
    # Try adding prefixes
    for prefix in MASK_PREFIXES:
        candidate = prefix + stem
        if candidate in mask_index:
            candidates.append((candidate, 2))
    
    return candidates


def match_pairs(
    image_files: List[Path],
    mask_files: List[Path],
    data_root: Path
) -> List[Tuple[Path, Path]]:
    """Match image files with corresponding mask files."""
    mask_index = index_by_stem(mask_files)
    pairs = []
    
    for img_file in image_files:
        candidates = get_mask_candidates(img_file.stem, mask_index)
        if candidates:
            best_stem = min(candidates, key=lambda x: x[1])[0]
            mask_file = mask_index[best_stem]
            pairs.append((img_file, mask_file))
    
    return sorted(pairs)


def write_split(
    pairs: List[Tuple[Path, Path]],
    train_ratio: float,
    runs_dir: Path,
    repo_root: Path,
    dry_run: bool = False
) -> Tuple[int, int]:
    """Split pairs and write to files."""
    train_count = int(len(pairs) * train_ratio)
    train_pairs = pairs[:train_count]
    test_pairs = pairs[train_count:]
    
    if not dry_run:
        runs_dir.mkdir(exist_ok=True)
        
        train_file = runs_dir / 'split_train.txt'
        with open(train_file, 'w') as f:
            for img, mask in train_pairs:
                img_rel = img.relative_to(repo_root)
                mask_rel = mask.relative_to(repo_root)
                f.write(f"{img_rel.as_posix()} {mask_rel.as_posix()}\n")
        
        test_file = runs_dir / 'split_test.txt'
        with open(test_file, 'w') as f:
            for img, mask in test_pairs:
                img_rel = img.relative_to(repo_root)
                mask_rel = mask.relative_to(repo_root)
                f.write(f"{img_rel.as_posix()} {mask_rel.as_posix()}\n")
    
    return len(train_pairs), len(test_pairs)


def main():
    parser = argparse.ArgumentParser(description='Dataset splitting for brain tumor images.')
    parser.add_argument('--data_root', default='data/brain_tumor', help='Path to dataset root')
    parser.add_argument('--runs_dir', default='runs', help='Output directory for split files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for deterministic split')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train split ratio')
    parser.add_argument('--dry_run', action='store_true', help='Print summary without writing files')
    
    args = parser.parse_args()
    
    repo_root = Path.cwd()
    data_root = repo_root / args.data_root
    runs_dir = repo_root / args.runs_dir
    
    if not data_root.exists():
        print(f"Error: Data root does not exist: {data_root}")
        return 1
    
    # Find files
    image_files = find_files(data_root, SUPPORTED_EXTS)
    mask_files = find_files(data_root, SUPPORTED_EXTS)
    
    if not image_files or not mask_files:
        print("Error: No image or mask files found.")
        if image_files:
            print(f"Example image stems: {[f.stem for f in image_files[:3]]}")
        if mask_files:
            print(f"Example mask stems: {[f.stem for f in mask_files[:3]]}")
        return 1
    
    # Match pairs
    pairs = match_pairs(image_files, mask_files, data_root)
    
    if not pairs:
        print("Error: No matched pairs found.")
        print(f"Example image stems: {[f.stem for f in image_files[:3]]}")
        print(f"Example mask stems: {[f.stem for f in mask_files[:3]]}")
        return 1
    
    # Shuffle deterministically
    random.seed(args.seed)
    random.shuffle(pairs)
    
    # Write split
    train_count, test_count = write_split(
        pairs, args.train_ratio, runs_dir, repo_root, args.dry_run
    )
    
    # Print summary
    print(f"Total images found: {len(image_files)}")
    print(f"Total masks found: {len(mask_files)}")
    print(f"Total matched pairs: {len(pairs)}")
    print(f"Train count: {train_count} ({100*train_count/len(pairs):.1f}%)")
    print(f"Test count: {test_count} ({100*test_count/len(pairs):.1f}%)")
    print("\nFirst 3 pairs:")
    for i, (img, mask) in enumerate(pairs[:3]):
        img_rel = img.relative_to(repo_root).as_posix()
        mask_rel = mask.relative_to(repo_root).as_posix()
        print(f"  {i+1}. {img_rel} {mask_rel}")
    
    if args.dry_run:
        print("\n[DRY RUN] No files written.")
    else:
        print(f"\nSplit files written to {runs_dir}/")
    
    return 0


if __name__ == '__main__':
    exit(main())