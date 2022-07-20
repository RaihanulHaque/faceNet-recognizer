import os
from os import listdir
from PIL import Image
from numpy import asarray
from numpy import expand_dims
from tensorflow import keras
import numpy as np
import pickle
import cv2


HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
MyFaceNet = keras.models.load_model('facenet_keras.h5')
folder = 'dataset/'
database = {}

for filename in listdir(folder):
    path = folder + filename
    img = cv2.imread(folder + filename)
    imgH = HaarCascade.detectMultiScale(img,1.1,4)
    if len(imgH)>0:
        x1, y1, width, height = imgH[0]         
    else:
        x1, y1, width, height = 1, 1, 10, 10 
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    
    imgP = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgP = Image.fromarray(imgP)                  # konversi dari OpenCV ke PIL
    img_array = asarray(imgP)
    face = img_array[y1:y2, x1:x2]                        
    
    face = Image.fromarray(face)                       
    face = face.resize((160,160))
    face = asarray(face)
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    face = expand_dims(face, axis=0)
    signature = MyFaceNet.predict(face)
    database[os.path.splitext(filename)[0]]=signature

myfile = open("data.pkl", "wb")
pickle.dump(database, myfile)
myfile.close()

print(database)