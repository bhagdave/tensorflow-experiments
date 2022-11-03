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
directory = os.fsencode('/home/dave/Projects/tensorflow/windscreens/images/windscreens_sorted/check/repair') 
f = open('repair-replace-transfer-prediction-repair.csv', 'w', newline='')
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
    
    classes = m.predict(image_tensor, batch_size =10)
    print(filename)
    print(classes)
    percentage =  classes[0][0]*100

    writer.writerow([str(filename), str("{:.2f}".format(classes[0][0])), str("{:.2f}".format(percentage)) + '%'])

f.close()

