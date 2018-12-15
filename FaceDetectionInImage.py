import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('/home/anmolmalhotra97/Desktop/openCV/haarcascade_frontalface_default.xml')
if face_cascade.empty():
   print('file couldnt load, give up!')
img = cv2.imread("/home/anmolmalhotra97/Desktop/image.jpg", 1)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
    gray_image, scaleFactor = 1.05, minNeighbors = 5)
#print(type(faces))
#print(faces)
for x, y, w, h in faces:
    img=cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
# img_1 = cv2.imread("/home/anmolmalhotra97/Desktop/Anmol.JPG", 0)
# print(img)
# print(type(img))
# print(img.shape)
resized_image=cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4)))
cv2.imshow('Anmol', resized_image)
cv2.waitKey(0)
# cv2.waitKey(2000)
cv2.destroyAllWindows()
