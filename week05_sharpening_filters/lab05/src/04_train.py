import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from src.unet_model import build_unet
from src.dataset import MedicalImageDataset

# Fix sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def iou_metric(y_true, y_pred):
    """
    Intersection over Union metric.
    y_true, y_pred: tensors of shape (batch, height, width, 1)
    """
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou


def dice_metric(y_true, y_pred):
    """
    Dice coefficient metric.
    y_true, y_pred: tensors of shape (batch, height, width, 1)
    """
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred)
    cardinality = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    
    dice = (2.0 * intersection + 1e-7) / (cardinality + 1e-7)
    return dice


def load_split(split_file):
    """Load image/mask paths from split file."""
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    with open(split_file, 'r') as f:
        paths = [line.strip() for line in f.readlines() if line.strip()]
    return paths


def create_dataset(image_paths, img_size, batch_size, shuffle=False):
    """Create tf.data.Dataset from image paths."""
    dataset_obj = MedicalImageDataset(image_paths, img_size)
    
    dataset = tf.data.Dataset.from_generator(
        lambda: dataset_obj,
        output_signature=(
            tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(img_size, img_size, 1), dtype=tf.float32)
        )
    )
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def main():
    parser = argparse.ArgumentParser(description='Train U-Net for medical image segmentation')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--img_size', type=int, default=256, help='Image size (square)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()
    
    # Ensure output directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs/tb', exist_ok=True)
    
    # Load split files
    train_split = 'runs/split_train.txt'
    test_split = 'runs/split_test.txt'
    
    train_paths = load_split(train_split)
    test_paths = load_split(test_split)
    
    print(f"Number of training samples: {len(train_paths)}")
    print(f"Number of validation samples: {len(test_paths)}")
    
    # Create datasets
    train_dataset = create_dataset(train_paths, args.img_size, args.batch_size, shuffle=True)
    val_dataset = create_dataset(test_paths, args.img_size, args.batch_size, shuffle=False)
    
    # Build model
    model = build_unet(
        input_shape=(args.img_size, args.img_size, 3),
        base_filters=32,
        depth=4
    )
    
    # Compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=args.lr),
        metrics=[iou_metric, dice_metric]
    )
    
    # Callbacks
    checkpoint_path = 'models/unet_best.keras'
    tb_logdir = 'runs/tb/'
    
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=tb_logdir,
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    print(f"Model save path: {os.path.abspath(checkpoint_path)}")
    print(f"TensorBoard log path: {os.path.abspath(tb_logdir)}")
    
    # Train
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )


if __name__ == '__main__':
    main()