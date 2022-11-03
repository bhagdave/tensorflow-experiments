import os
import argparse
import csv
import urllib.request as urllib
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np


image_height, image_width = 300, 300
m = load_model('/home/dave/Projects/tensorflow/windscreens/repair-replace-transfer')
directory = os.fsencode('/home/dave/Projects/tensorflow/windscreens/images/resizing') 
f = open('replace-predictions-transfer-resizing.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerow(['filename', 'classification', 'percentage1', 'percentage2'])
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
    percentage2 =  classes[1][0]*100

    writer.writerow([str(filename), str("{:.2f}".format(classes[0][0])), str("{:.2f}".format(percentage1)), str("{:.2f}".format(percentage1))])

f.close()

