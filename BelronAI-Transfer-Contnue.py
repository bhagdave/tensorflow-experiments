import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import random
import json
from SharedClasses import f1_score, CustomEarlyStopping, CustomImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import DepthwiseConv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, Add, Activation, Input, Dense, Flatten
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import HeUniform


image_folder = './images-new/close_up'
image_height = 224
image_width = 224
model_name = 'repair-replace-cross'
batch_size = 8
num_classes = 2
learning_rate = 0.01
dropout_rate1 = 0.1
dropout_rate2 = 0.3
regularisation_rate = 0.0001
early_stopping_patience = 10
num_epochs = 100
dense_layer_size = 1024


# Initialize the CustomImageDataGenerator for training and validation
train_generator = CustomImageDataGenerator(os.path.join(image_folder, 'train/'), image_width, image_height, batch_size=batch_size)
validation_generator = CustomImageDataGenerator(os.path.join(image_folder, 'validate/'), image_width, image_height, batch_size=batch_size)

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(image_height, image_width, 3))

# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

# Create the model
input_tensor = Input(shape=(image_height, image_width, 3))
x = base_model(input_tensor)
residual = x
x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)  # Batch normalization before activation
x = Activation('relu')(x)

# Add the residual (original input) to the output of the above layer/block
x = Add()([x, residual])
x = Activation('relu')(x)  # Optional: Apply activation after adding the residual
x = Flatten()(x)  # Flatten the output
x = BatchNormalization()(x)
x = Dropout(dropout_rate1)(x)  # Apply dropout
x = Dense(dense_layer_size, activation='relu', kernel_regularizer=regularizers.l2(regularisation_rate))(x)  # Add a dense layer
x = Dropout(dropout_rate2)(x)  # Apply dropout again
predictions = Dense(num_classes, activation='softmax')(x)  # Final layer with softmax activation for classification

model = Model(inputs=input_tensor, outputs=predictions)

rmsprop_optimizer = Adam(learning_rate=learning_rate)

def scheduler(epoch, lr):
    if epoch < 20:
        return learning_rate
    elif epoch < 40:
        return learning_rate * .5
    elif epoch < 60:
        return learning_rate * .1
    else:
        return learning_rate * .05

model.load_weights(f"models/{model_name}.keras")
model.compile(optimizer=rmsprop_optimizer, loss='categorical_crossentropy', metrics=['accuracy', f1_score])

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,patience=8, min_lr=0.0001)

checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_accuracy', save_best_only=True, mode='auto')
# Define the early stopping criteria
early_stopping_loss = EarlyStopping(monitor='val_loss',verbose=1, patience=early_stopping_patience, mode='min')
early_stopping_accuracy = EarlyStopping(monitor='val_accuracy', min_delta=0.001,verbose=1, patience=early_stopping_patience, mode='max')
early_stopping_f1 = EarlyStopping(monitor='val_f1_score',min_delta=0.001,verbose=1, patience=early_stopping_patience, mode='max')
learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
model.summary()
model.fit(
    x=train_generator.generate_data(),
    epochs=num_epochs,
    steps_per_epoch=train_generator.calculate_num_samples() // train_generator.batch_size,
    validation_data=validation_generator.generate_data(),
    validation_steps=validation_generator.calculate_num_samples() // validation_generator.batch_size,
    verbose=1,
    callbacks=[early_stopping_loss, checkpoint, reduce_lr, learning_rate_callback, early_stopping_f1, early_stopping_accuracy]
)

# Save the model
print(f"Saving {model_name}")
model.save(f"models/{model_name}.keras")

