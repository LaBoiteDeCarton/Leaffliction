import tensorflow as tf
seed = 117

def get_augmentation():
    # Creating data augmentation pipeline
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