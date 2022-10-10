# FaceNet - Facial Recognizer

Foobar is a facial recognization program using Google's FaceNet.

## Installation
First you need to install Python 3.8. So go to this official [Link](https://www.python.org/downloads/release/python-3810) to download and install it for Windows operating system. Use this link to download [FaceNet Keras Model](shorturl.at/nLM05).
Now use this bellow commands in command prompt to install tensorflow, opencv, numpy, pillow and pyfirmata to run this project.
```bash
pip install tensorflow==2.3.0
pip install opencv-contrib-python==4.4.0.46
pip install numpy
pip install Pillow==7.2.0
pip install pyfirmata
```
Then install the Arduino IDE

## Basic setup
Now open Visual Studio code and look in the extensions for “Python & Code Runner”. After installing these two, open the downloaded codebase from this repository. Then add the FaceNet model in the codebase directory. After that, there is a folder named dataset in the codebase. Then just put the pictures you need to train in that dataset folder.

Then open Arduino IDE and go to File -> Examples -> Firmata -> StandardFirmata. After that upload this StandardFirmata file to Arduino.

Then there is a file labeled as traindata.py and after putting those images you will run this python file. This will create a pkl file named data.pkl which is a numpy database and you don't need to worry about this file because those trained images are stored in this file as numpy array.

After that, just run the faceRecogniser.py file and voila... it works....

When this code will run it will start looking for faces from the camera and start predicting these faces with the help of the facenet algorithm. When it matches some face with trained ones it will open the security door and after a few seconds, it'll automatically close the door and write their name and their log time in an excel/CSV file named data.csv

In both trainData.py and faceRecogniser.py, both haarcascade's xml file and trained keras model is loaded into two variables named HaarCascade and MyFaceNet
```python
from tensorflow import keras
import cv2

HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
MyFaceNet = keras.models.load_model('facenet_keras.h5')
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
