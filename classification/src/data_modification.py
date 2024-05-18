import tensorflow as tf
seed = 117

def get_modification():
    """
    This function creates a Sequential model in TensorFlow with three layers of image augmentations:
    1. RandomRotation: Rotates the image by a random factor within the range (-0.25, 0.3).
    2. RandomBrightness: Adjusts the brightness of the image by a random factor within the range (-0.45, 0.45).
    3. RandomContrast: Adjusts the contrast of the image by a factor of 0.5.
    
    The seed for the random number generator is set to ensure reproducibility of the augmentations.
    
    Returns:
        tf.keras.Sequential: A Sequential model with the specified image augmentation layers.
    """
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(
            factor = (-.25, .3),
            fill_mode = 'reflect',
            interpolation = 'bilinear',
            seed = seed),
            
            
            tf.keras.layers.RandomBrightness(
            factor = (-.45, .45),
            value_range = (0.0, 1.0),
            seed = seed),
            
            tf.keras.layers.RandomContrast(
            factor = (.5),
            seed = seed)
        ]
    )
    return augmentation