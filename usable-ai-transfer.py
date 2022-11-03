import tensorflow as tf
import tensorflow_hub as hub
import urllib.request
import os
import keras_tuner as kt
from tensorflow.keras import layers
from tensorflow.keras import Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, RandomFlip, RandomRotation
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
import keras_tuner as kt

def model_builder(hp):
    weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    weights_file = "inception_v3.h5"
    urllib.request.urlretrieve(weights_url, weights_file)
    pre_trained_model = InceptionV3(input_shape=(300, 300, 3),
                                    include_top=False,
                                    weights=None)

    pre_trained_model.load_weights(weights_file)

    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.get_layer('mixed7')
    last_output = last_layer.output
    x = Flatten()(last_output)
    hp_dense1 = hp.Int('dense1', min_value=64, max_value=1024, step=32)
    x = layers.Dense(hp_dense1, activation='relu')(x)
    x = layers.Dense(2, activation='softmax')(x)
    model = Model(pre_trained_model.input, x)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=hp_learning_rate), metrics=['accuracy'])
    return model

tuner = kt.Hyperband(model_builder, objective='val_accuracy',max_epochs=15, factor=3, directory='tuner',project_name='usable-ai-transfer')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

image_folder = '/home/dave/Projects/tensorflow/windscreens/usable-ai'
image_height = 300
image_width = 300
model_name = 'usable-transfer'

train_datagen = ImageDataGenerator(
        rescale=1./255, 
        horizontal_flip=True, 
        vertical_flip=True,
        rotation_range=40,
        width_shift_range=0.2,
        zoom_range=0.2
)
train_generator = train_datagen.flow_from_directory(
            '/home/dave/Projects/tensorflow/windscreens/usable-ai/train',
            target_size=(image_width, image_height),
            batch_size=100,
            class_mode='categorical'
)
test_datagen = ImageDataGenerator(
        rescale=1./255, 
)
test_generator = train_datagen.flow_from_directory(
            '/home/dave/Projects/tensorflow/windscreens/usable-ai/test',
            target_size=(image_width, image_height),
            batch_size=100,
            class_mode='categorical'
)

tuner.search(train_generator,epochs=15, validation_data=test_generator, callbacks=[stop_early])
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_generator,epochs=50,validation_data=test_generator)
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print ('Best epoch:' + str( best_epoch))
hypermodel = tuner.hypermodel.build(best_hps)
history = hypermodel.fit(train_generator,epochs=best_epoch, validation_data=test_generator)
hypermodel.save(model_name)


