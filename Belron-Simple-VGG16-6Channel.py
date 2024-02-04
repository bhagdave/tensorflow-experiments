import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import random
import json
from SharedClasses import f1_score, CustomEarlyStopping, CustomImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import DepthwiseConv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import numpy as np
from PIL import Image
os.environ['TF_GRPC_TIMEOUT'] = '3600'  # Set it to 1 hour (3600 seconds)

image_folder = './images-for-prediction'
image_height = 300
image_width = 300
model_name = 'belron-simple'
batch_size = 8
num_classes = 2
num_epochs = 20 
conv_1_units = 160
dropout_rate = 0.2
dense_1_units = 160 
dense_2_units = 192 
dense_3_units = 192
dense_4_units = 96
early_stopping = 3
steps_per_epoch = 100
learning_rate = 0.0001
validation_steps = 20

def scheduler(epoch, lr):
    if epoch < 50:
        return learning_rate
    elif epoch < 75:
        return learning_rate
    elif epoch < 100:
        return learning_rate
    else:
        return learning_rate


# Define your custom condition function
def custom_condition(logs):
    # You can define your condition based on loss, accuracy, or any other metric
    return logs.get('val_loss') < 0.05  # Example: Stop if loss is less than 0.2

# Create the custom callback
custom_early_stopping = CustomEarlyStopping(condition=custom_condition, verbose=1)

# Initialize the CustomImageDataGenerator for training and validation
train_generator = CustomImageDataGenerator(os.path.join(image_folder, 'train/'), image_width, image_height, batch_size=batch_size)
validation_generator = CustomImageDataGenerator(os.path.join(image_folder, 'validate/'), image_width, image_height, batch_size=batch_size)

def model_builder(input_shape=(300, 300, 6)):
    print("Building model")
    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Dropout(0.5, name='dropout')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='dense')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    outputs = Dense(2, activation='softmax', name='dense_1')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='custom_vgg16_model')

    return model

model = model_builder()
model.summary()
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy', f1_score]
)

checkpoint = ModelCheckpoint('model-{epoch:03d}.keras', monitor='val_accuracy', save_best_only=True, mode='auto')
# Define the early stopping criteria
early_stopping = EarlyStopping(monitor='val_accuracy', patience=early_stopping, verbose=1, mode='auto', restore_best_weights=True)
learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# model.fit_generator(train_generator, epochs=num_epochs, validation_data=validation_generator, callbacks=[early_stopping, learning_rate_callback])

print("Training model")
# Fit the model
model.fit(
    train_generator.generate_data(),
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator.generate_data(),
    validation_steps=validation_steps,
    verbose=1,
    callbacks=[learning_rate_callback, checkpoint, custom_early_stopping],
)

# Save the model
model.save(f"{model_name}.keras")

