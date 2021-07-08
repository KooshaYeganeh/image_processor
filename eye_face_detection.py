import cv2
import numpy as np
face_cascade =cv2.CascadeClassifier('./Cascade/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('./Cascade/haarcascade_eye.xml')
eye_glass = cv2.CascadeClassifier('./Cascade/haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)


    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_grey = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_grey)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)


        eyes = eye_glass.detectMultiScale(roi_grey)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,255),2)


    cv2.imshow('face', frame)
    c = cv2.waitKey(27) & 0xFF
    if c == 27:
        break

cv2.destroyAllWindows()
cap.release()
