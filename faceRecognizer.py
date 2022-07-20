from PIL import Image
from tensorflow import keras
import numpy as np
from numpy import asarray
from numpy import expand_dims
from datetime import datetime
import csv
import pickle
import cv2
from pyfirmata import Arduino, SERVO



def writeLog(Name):
    file = open("data.csv","a+")
    csvwrite = csv.writer(file)
    # fieldName = ['Name','Time']
    # csvwrite.writerow(fieldName)
    # users = ['Rahi','Anan','Sir']
    csvwrite.writerow([Name,datetime.now()])
    file.close()

HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
MyFaceNet = keras.models.load_model('facenet_keras.h5')

board = Arduino("/dev/ttyACM0")
board.digital[9].mode = SERVO
INITIALANGLE = 0

myfile = open("data.pkl", "rb")
database = pickle.load(myfile)
myfile.close()

temporary = ""

# cap = cv2.VideoCapture('https://192.168.0.100:8080/video')
cap = cv2.VideoCapture(0)
file = open("data.csv","a+")
csvwrite = csv.writer(file)
fieldName = ['Name','Time']
csvwrite.writerow(fieldName)

file.close()
while True:
    _, img = cap.read()
    # img = cv2.imread('test2.jpeg')
    img = cv2.resize(img, (360,440))
    
    imgH = HaarCascade.detectMultiScale(img,1.1,4)
    
    if len(imgH)>0:
        x1, y1, width, height = imgH[0]        
    else:
        x1, y1, width, height = 1, 1, 10, 10
    
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    
    
    imgP = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgP = Image.fromarray(imgP)                  # Converts OPENCV to PIL
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
    
    min_dist=100
    identity=' '
    for key, value in database.items() :
        dist = np.linalg.norm(value-signature)
        if dist < min_dist:
            min_dist = dist
            identity = key

    if identity[:4] != temporary and identity[:4] != 'None':
        writeLog(identity)
        temporary = identity
        board.digital[9].write(90)
        
    cv2.putText(img,identity[:4], (100,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.rectangle(img,(x1,y1),(x2,y2), (0,255,0), 2)
    print(identity[:4]," ",min_dist)
    cv2.imshow('res',img)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
        
# cv2.destroyAllWindows()
# cap.release()