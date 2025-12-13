import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def conv_block(x, filters, kernel_size=3, activation="relu", use_batchnorm=False, dropout=0.0, name=None):
    """
    Double convolution block with optional batch normalization and dropout.
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Kernel size (default 3)
        activation: Activation function (default "relu")
        use_batchnorm: Whether to use BatchNormalization (default False)
        dropout: Dropout rate (default 0.0, no dropout)
        name: Optional name prefix for layers
    
    Returns:
        Output tensor after 2x Conv2D blocks
    """
    conv_name_1 = f"{name}_conv1" if name else None
    conv_name_2 = f"{name}_conv2" if name else None
    bn_name_1 = f"{name}_bn1" if name else None
    bn_name_2 = f"{name}_bn2" if name else None
    dropout_name = f"{name}_dropout" if name else None
    
    # First convolution
    x = layers.Conv2D(filters, kernel_size, padding="same", name=conv_name_1)(x)
    if use_batchnorm:
        x = layers.BatchNormalization(name=bn_name_1)(x)
    x = layers.Activation(activation)(x)
    
    # Second convolution
    x = layers.Conv2D(filters, kernel_size, padding="same", name=conv_name_2)(x)
    if use_batchnorm:
        x = layers.BatchNormalization(name=bn_name_2)(x)
    x = layers.Activation(activation)(x)
    
    # Optional dropout
    if dropout > 0.0:
        x = layers.Dropout(dropout, name=dropout_name)(x)
    
    return x


def encoder_block(x, filters, pool_size=(2, 2), **conv_kwargs):
    """
    Encoder block: convolution followed by max pooling.
    
    Args:
        x: Input tensor
        filters: Number of filters
        pool_size: Max pooling size (default (2, 2))
        **conv_kwargs: Additional arguments for conv_block
    
    Returns:
        Tuple of (skip_connection, pooled_output)
    """
    skip = conv_block(x, filters, **conv_kwargs)
    pooled = layers.MaxPooling2D(pool_size)(skip)
    return skip, pooled


def decoder_block(x, skip, filters, up_mode="transpose", **conv_kwargs):
    """
    Decoder block: upsampling, concatenation with skip connection, and convolution.
    
    Args:
        x: Input tensor (from previous decoder level)
        skip: Skip connection from corresponding encoder level
        filters: Number of filters
        up_mode: Upsampling mode - "transpose" (Conv2DTranspose) or "upsample" (UpSampling2D + Conv2D)
        **conv_kwargs: Additional arguments for conv_block
    
    Returns:
        Output tensor after decoding
    """
    if up_mode == "transpose":
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(x)
    else:  # "upsample"
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(filters, (2, 2), padding="same")(x)
    
    # Concatenate with skip connection
    x = layers.Concatenate(axis=-1)([x, skip])
    
    # Convolution block
    x = conv_block(x, filters, **conv_kwargs)
    
    return x


def build_unet(input_shape, base_filters=32, depth=4, dropout=0.0, use_batchnorm=False, up_mode="transpose"):
    """
    Build a U-Net model for binary semantic segmentation.
    
    Args:
        input_shape: Input shape (H, W, C) without batch dimension
        base_filters: Number of filters at the first encoder level (default 32)
        depth: Depth of the U-Net (number of encoder/decoder levels, default 4)
        dropout: Dropout rate (default 0.0)
        use_batchnorm: Whether to use batch normalization (default False)
        up_mode: Upsampling mode - "transpose" or "upsample" (default "transpose")
    
    Returns:
        Compiled Keras Model
    """
    inputs = keras.Input(shape=input_shape, name="input")
    x = inputs
    
    # Encoder
    skips = []
    conv_kwargs = {
        "activation": "relu",
        "use_batchnorm": use_batchnorm,
        "dropout": dropout
    }
    
    for level in range(depth):
        filters = base_filters * (2 ** level)
        skip, x = encoder_block(x, filters, name=f"encoder_L{level}", **conv_kwargs)
        skips.append(skip)
    
    # Bottleneck
    bottleneck_filters = base_filters * (2 ** depth)
    x = conv_block(x, bottleneck_filters, name="bottleneck", **conv_kwargs)
    
    # Decoder
    for level in range(depth - 1, -1, -1):
        filters = base_filters * (2 ** level)
        skip = skips[level]
        x = decoder_block(x, skip, filters, up_mode=up_mode, name=f"decoder_L{level}", **conv_kwargs)
    
    # Output layer: binary segmentation (sigmoid activation)
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", name="output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="UNet")
    
    # Compile with appropriate loss for binary segmentation
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["dice_coefficient"]
    )
    
    return model


if __name__ == "__main__":
    # Test the model with example input shape
    input_shape = (256, 256, 3)
    model = build_unet(input_shape, base_filters=32, depth=4, dropout=0.0, use_batchnorm=False)
    
    print("=" * 80)
    print("U-Net Model Summary")
    print("=" * 80)
    model.summary()
    
    print("\n" + "=" * 80)
    print("Output Shape Verification")
    print("=" * 80)
    output_shape = model.output_shape
    print(f"Input shape:  {model.input_shape}")
    print(f"Output shape: {output_shape}")
    
    # Assert output has 1 channel for binary segmentation
    assert output_shape[-1] == 1, f"Expected output channels = 1, got {output_shape[-1]}"
    print("✓ Output shape is correct: (None, 256, 256, 1)")
    print("=" * 80)