import tensorflow as tf
import os
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, concatenante, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, RandomFlip, RandomRotation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K

image_folder = '/home/dave/Projects/tensorflow/tensorflow-experiments/images'
image_height = 300
image_width = 300
model_name = 'repair-replace-cross'

class TwoImageGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size=32):
        self.directory = directory
        self.batch_size = batch_size
        self.image_files = os.listdir(directory)
        self.indexes = np.arrange(len(self.image_files) // 2)

    def __len__(self):
        return (len(self.indexes)

    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [image_height, image_width])
        image /= 255.0
        return image

    def create_image_pairs(folder_path):
        files = os.listdir(folder_path)
        paired_files = {}

        for file in files:
            job_id = file.split('_', 1)
            if job_id not in paired_files:
                paired_files[job_id] = []
            paired_files[job_id].append(os.path.join(folder_path, file))

        return [tuple(paired_files[job_id]) for job_id in paired_files if len(paired_files[job_id]) == 2]

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_1  = []
        batch_2  = []
        labels = []
        for idx in indexes:
            img_name_1 = self.image_files[2*idx]
            img_name_2 = self.image_files[2*idx+1]
            # load and preprocess images
            batch_1.append(img_1)
            batch_2.append(img_2)
            labels.append(label)
        return [np.array(batch_1), np.array(batch_2)], np.array(labels)


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

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
    model.add(Dropout(0.5))
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
            batch_size=10,
            class_mode='categorical'
)
test_datagen = ImageDataGenerator(
        rescale=1./255, 
)
test_generator = test_datagen.flow_from_directory(
            os.path.join(image_folder, 'validate/'),
            target_size=(image_width, image_height),
            batch_size=10,
            class_mode='categorical'
)


model = model_builder()
model.summary()
model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy', f1_score]
)
model.fit(train_generator,epochs=44, validation_data=test_generator, verbose=1)
model.save(f"{model_name}.h5")
