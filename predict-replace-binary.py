import os
import argparse
import csv
import urllib.request as urllib
from keras.models import load_model
import keras.utils as image
import cv2
import numpy as np


image_height, image_width = 300, 300
m = load_model('/home/dave/Projects/tensorflow/tensorflow-experiments/repair-replace-binary')
directory = os.fsencode('/home/dave/Projects/tensorflow/tensorflow-experiments/images/test/replace') 
f = open('replace-binary.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerow(['filename', 'classification', 'percentage'])
count = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    image_path = os.path.join(os.fsdecode(directory), filename)
    img = image.load_img(image_path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    image_tensor = np.vstack([x])
    
    classes = m.predict(image_tensor)
    percentage1 =  classes[0][0]*100
    print(classes)

    writer.writerow([str(filename), str("{:.2f}".format(classes[0][0])), str("{:.2f}".format(percentage1))])

f.close()
