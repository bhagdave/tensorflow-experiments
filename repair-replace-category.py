import tensorflow as tf
import os
import keras_tuner as kt
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, RandomFlip, RandomRotation
from tensorflow.keras.optimizers import RMSprop

image_folder = '/home/dave/Projects/tensorflow/windscreens/images/windscreens_sorted'
image_height = 300
image_width = 300
model_name = 'repair-replace-cross'

def model_builder():
    model = Sequential()
    model.add(Conv2D(112, (3,3), activation='relu', input_shape=(300, 300, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(192, (3,3), activation='relu') )
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(96, (3,3), activation='relu') )
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(160, (3,3), activation='relu') )
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(96, (3,3), activation='relu') )
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(112, activation='relu'))
    model.add(Dense(240, activation='relu'))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model


train_datagen = ImageDataGenerator(
        rescale=1./255, 
        horizontal_flip=True, 
        vertical_flip=True,
        rotation_range=40,
        width_shift_range=0.2,
        zoom_range=0.2
)
train_generator = train_datagen.flow_from_directory(
            os.path.join(image_folder, 'train/'),
            target_size=(image_width, image_height),
            batch_size=128,
            class_mode='categorical'
)
test_datagen = ImageDataGenerator(
        rescale=1./255, 
)
test_generator = train_datagen.flow_from_directory(
            os.path.join(image_folder, 'validate/'),
            target_size=(image_width, image_height),
            class_mode='categorical'
)


model = model_builder()
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_generator,epochs=44, validation_data=test_generator, verbose=1)
model.save(model_name)
