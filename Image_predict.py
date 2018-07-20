import numpy as np
import pandas as pd
import pickle
import cv2
import glob

face_cascade = cv2.CascadeClassifier('C:/opencv-3.4.1/data/haarcascades/haarcascade_frontalface_default.xml')

#loading model
model = pickle.load(open('Gender_model.sav', 'rb'))

#detecting faces in images
num = 0
for file in glob.glob("try/*"):
    try:
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if faces == ():
            print(file, " No faces found")
        m = 0
        f = 0
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img ,(x,y),(x+w,y+h),(255,0,0),2)
            dim = (20,20)
            resize_img = img[y:y+h,x:x+h].copy()
            resized = cv2.resize(resize_img, dim, interpolation = cv2.INTER_AREA)
            gray2 = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            a = gray2.ravel()
            b = a.reshape(1, 400)
            data = pd.DataFrame(b, columns = [str(l) for l in range(0,400)])

            result = model.predict(data)
            probab = model.predict_proba(data)
            num+=1
            if result == [1]:
                m+=1
                print(file, "Male", probab)
            else:
                f+=1
                print(file, " Female", probab)
        print("There are a total of", m, "males and", f, "females in image")
        print(" ")
    except:
        print(file, " No faces found")

print("There are a total of", num, "persons in dataset")
