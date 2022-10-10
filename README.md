# FaceNet - Facial Recognizer

Foobar is a facial recognization program using Google's FaceNet.

## Installation
First you need to install Python 3.8. So go to this official [Link](https://www.python.org/downloads/release/python-3810) to download and install it for Windows operating system.
Use this link to download [FaceNet Keras Model](https://drive.google.com/file/d/19pVoy7iikeOec5YmI3Vglr-gNrD8Ae4V/view?usp=sharing).
Now use this bellow commands in command prompt to install tensorflow, opencv, numpy, pillow and pyfirmata to run this project.
```bash
pip install tensorflow==2.3.0
pip install opencv-contrib-python==4.4.0.46
pip install numpy
pip install Pillow==7.2.0
pip install pyfirmata
```

## Basic setup
Now open Visual Syudio code and look in extension for "Python & Code Runner". After installing them open the downloaded codebase from this repository.

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
