import tensorflow as tf
import os
import keras_tuner as kt
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, RandomFlip, RandomRotation
from tensorflow.keras.optimizers import Adam

image_folder = '/home/dave/Projects/tensorflow/tensorflow-experiments/images'
image_height = 300
image_width = 300
model_name = 'repair-replace'

def model_builder(hp):
    model = Sequential()
    hp_layer1 = hp.Int('conv1', min_value=16, max_value=256, step=16)
    model.add(Conv2D(hp_layer1, (3,3), activation='relu', input_shape=(300, 300, 3)))
    model.add(MaxPooling2D(2, 2))
    hp_layer2 = hp.Int('conv2', min_value=16, max_value=256, step=16)
    model.add(Conv2D(hp_layer2, (3,3), activation='relu') )
    model.add(MaxPooling2D(2, 2))
    hp_layer3 = hp.Int('conv3', min_value=16, max_value=256, step=16)
    model.add(Conv2D(hp_layer3, (3,3), activation='relu') )
    model.add(MaxPooling2D(2, 2))
    hp_layer4 = hp.Int('conv4', min_value=16, max_value=256, step=16)
    model.add(Conv2D(hp_layer4, (3,3), activation='relu') )
    model.add(MaxPooling2D(2, 2))
    hp_layer5 = hp.Int('conv5', min_value=16, max_value=256, step=16)
    model.add(Conv2D(hp_layer5, (3,3), activation='relu') )
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    hp_dense1 = hp.Int('dense1', min_value=16, max_value=512, step=32)
    model.add(Dense(hp_dense1, activation='relu'))
    hp_dense2 = hp.Int('dense2', min_value=16, max_value=256, step=32)
    model.add(Dense(hp_dense2, activation='relu'))
    hp_dense3 = hp.Int('dense3', min_value=16, max_value=128, step=16)
    model.add(Dense(hp_dense3, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=hp_learning_rate), metrics=['accuracy'])
    return model

tuner = kt.Hyperband(model_builder, objective='val_accuracy',max_epochs=15, factor=3, directory='tuner',project_name='repair-ai')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

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
            batch_size=10,
            class_mode='categorical'
)
test_datagen = ImageDataGenerator(
        rescale=1./255, 
)
test_generator = train_datagen.flow_from_directory(
            os.path.join(image_folder, 'validate/'),
            batch_size=10,
            target_size=(image_width, image_height),
            class_mode='categorical'
)


tuner.search(train_generator,epochs=25, validation_data=test_generator, callbacks=[stop_early])
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_generator,epochs=50,validation_data=test_generator)
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print ('Best epoch:' + str( best_epoch))
hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(train_generator,epochs=best_epoch, validation_data=test_generator)
hypermodel.save(model_name)
