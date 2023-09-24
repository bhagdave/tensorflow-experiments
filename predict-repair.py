import os
import argparse
import csv
import urllib.request as urllib
from keras.models import load_model
import keras.utils as image
import cv2
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

image_height, image_width = 300, 300
m = load_model('/home/dave/Projects/tensorflow/tensorflow-experiments/repair-replace-cross.h5', custom_objects={'f1_score': f1_score})
directory = os.fsencode('/home/dave/Projects/tensorflow/tensorflow-experiments/images/test/repair') 
f = open('repair-category.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerow(['filename', 'classification', 'percentage'])
count = 0

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    image_path = os.path.join(os.fsdecode(directory), filename)
    img = image.load_img(image_path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = x /255.0
    x = np.expand_dims(x, axis=0)
    image_tensor = np.vstack([x])
    
    classes = m.predict(image_tensor)
    percentage =  classes[0][0]*100
    print(classes)

    writer.writerow([str(filename), str("{:.2f}".format(classes[0][0])), str("{:.2f}".format(percentage)) + '%'])

f.close()

