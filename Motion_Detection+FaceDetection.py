import numpy as np
import cv2
import time
a = 1
face_cascade = cv2.CascadeClassifier('/home/anmolmalhotra97/Desktop/openCV/haarcascade_frontalface_default.xml')
if face_cascade.empty():
       print('file couldnt load, give up!')
first_frame = None
video = cv2.VideoCapture(0)
while True:
    a=a+1  
    check, frame = video.read()
    #print(frame)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    if first_frame is None:
        first_frame = gray
        continue
    delta_frame = cv2.absdiff(first_frame,gray)
    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta, None, iterations=0 )
    (_,cnts,_) = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),3)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors = 5)
    for x, y, w, h in faces:
        frame=cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0 , 0), 3)
    cv2.imshow('frame',frame)
    cv2.imshow('Capturing',gray)
    cv2.imshow('delta',delta_frame)
    cv2.imshow('thrash',thresh_delta)
    key = cv2.waitKey(1)
    if key==ord('q'):
        break
print(a)
video.release()
cv2.destroyAllWindows()