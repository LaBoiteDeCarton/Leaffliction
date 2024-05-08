import tensorflow as tf
from keras.layers import Rescaling # This layer rescales pixel values

train_dir = '../datasets/Train'
test_dir = '../datasets/Test'
val_dir = '../datasets/Validation'
seed = 117


def get_preprocessing():

    # Creating a Dataset for the Training data
    train = tf.keras.utils.image_dataset_from_directory(
        train_dir,  # Directory where the Training images are located
        labels = 'inferred', # Classes will be inferred according to the structure of the directory
        label_mode = 'categorical',
        class_names = ['Healthy', 'Powdery', 'Rust'],
        batch_size = 16,    # Number of processed samples before updating the model's weights
        image_size = (256, 256), # Defining a fixed dimension for all images
        shuffle = True,  # Shuffling data
        seed = seed,  # Random seed for shuffling and transformations
        validation_split = 0, # We don't need to create a validation set from the training set
        crop_to_aspect_ratio = True # Resize images without aspect ratio distortion
    )

    # Creating a dataset for the Test data
    test = tf.keras.utils.image_dataset_from_directory(
        test_dir,  
        labels = 'inferred', 
        label_mode = 'categorical',
        class_names = ['Healthy', 'Powdery', 'Rust'],
        batch_size = 16,    
        image_size = (256, 256), 
        shuffle = True,  
        seed = seed,  
        validation_split = 0, 
        crop_to_aspect_ratio = True 
    )

    # Creating a dataset for the Test data
    validation = tf.keras.utils.image_dataset_from_directory(
        val_dir,  
        labels = 'inferred', 
        label_mode = 'categorical',
        class_names = ['Healthy', 'Powdery', 'Rust'],
        batch_size = 16,    
        image_size = (256, 256),
        shuffle = True,  
        seed = seed,  
        validation_split = 0, 
        crop_to_aspect_ratio = True 
    )

    print('\nTraining Dataset:', train)
    print('\nTesting Dataset:', test)
    print('\nValidation Dataset:', validation)

    # Checking minimum and maximum pixel values in the Validation dataset
    min_value = float('inf')
    max_value = -float('inf')

    for img, label in validation:
        batch_min = tf.reduce_min(img)
        batch_max = tf.reduce_max(img)
        
        min_value = min(min_value, batch_min.numpy())
        max_value = max(max_value, batch_max.numpy())
        
    print('\nMinimum pixel value in the Validation dataset', min_value)
    print('\nMaximum pixel value in the Validation dataset', max_value)

    scaler = Rescaling(1./255) # Defining scaler values between 0 to 1

    # Rescaling datasets
    train = train.map(lambda x, y: (scaler(x), y)) 
    test = test.map(lambda x, y: (scaler(x), y))
    validation = validation.map(lambda x, y: (scaler(x), y))

    # Checking minimum and maximum pixel values in the Validation dataset
    min_value = float('inf')
    max_value = -float('inf')

    for img, label in validation:
        batch_min = tf.reduce_min(img)
        batch_max = tf.reduce_max(img)
        
        min_value = min(min_value, batch_min.numpy())
        max_value = max(max_value, batch_max.numpy())
        
    print('\nMinimum pixel value in the Validation dataset', min_value)
    print('\nMaximum pixel value in the Validation dataset', max_value)

    return train, test, validation