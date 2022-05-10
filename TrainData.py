import cv2
import numpy as np
import os
from PIL import Image

# De train dc anh can lay dc ID va lay dc 1 mang du lieu anh
# ID lay chinh tu duong dan cua anh

# thu vien mac dinh cua opencv dung de train hinh anh
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataSet'

def getImageWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        Id = int(imagePath.split('\\')[1].split('.')[1])
        faces.append(faceNp)
        Ids.append(Id)
        cv2.imshow('trainning', faceNp)
        cv2.waitKey(10)
    return faces, Ids

faces, Ids = getImageWithID(path)
recognizer.train(faces, np.array(Ids)) #Train. Sau khi train se tra ve 1 file dang yml
if not os.path.exists('recognizer'):
    os.makedirs('recognizer')
recognizer.save('recognizer/trainingData.yml')

cv2.destroyAllWindows()