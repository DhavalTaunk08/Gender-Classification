import numpy as np
import cv2
import csv
import glob
import pandas as pd
import os

face_cascade = cv2.CascadeClassifier('C:/opencv-3.4.1/data/haarcascades/haarcascade_frontalface_default.xml')

data = pd.DataFrame()
c=0
for file in glob.glob("data_male/*"):
    try:
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img ,(x,y),(x+w,y+h),(255,0,0),2)
            dim = (20,20)
            resize_img = img[y:y+h,x:x+h].copy()
            resized = cv2.resize(resize_img, dim, interpolation = cv2.INTER_AREA)
            gray2 = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            a = gray2.ravel()
            b = a.reshape(1,400)
            df = pd.DataFrame(b, columns = [str(k) for k in range(0,400)])
            df['400'] = 1
            frames = [data, df]
            data = pd.concat(frames, ignore_index=True)
    except:
        c+=1

d = 0
for file in glob.glob("data_female/*"):
    try:
        img = cv2.imread(file, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img ,(x,y),(x+w,y+h),(255,0,0),2)
            dim = (20,20)
            resize_img = img[y:y+h,x:x+h].copy()
            resized = cv2.resize(resize_img, dim, interpolation = cv2.INTER_AREA)
            gray2 = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            a = gray2.ravel()
            b = a.reshape(1, 400)
            df = pd.DataFrame(b, columns = [str(l) for l in range(0,400)])
            #df = pd.join(mt)
            df['400'] = 0
            frames = [data, df]
            data = pd.concat(frames, ignore_index = True)
    except:
        d+=1

print(data)
data.to_csv('dataset.csv', sep= ',')
print(c)
print(d)
