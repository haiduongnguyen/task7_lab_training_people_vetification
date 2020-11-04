import cv2
import numpy as np
import pandas as pd
from get_weight import predict

video_path = 'resource/video_task_7.mp4'

cap = cv2.VideoCapture(video_path)
backsub = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=None, detectShadows=False)
ret = True

# some para for con tour
x,y,w,h = 0,0,0,0

while ret:
    ret, framecolor = cap.read()
    frame = cv2.cvtColor(framecolor, cv2.COLOR_BGR2GRAY)
    frame_sub = backsub.apply(frame)
    contours, hierarchy= cv2.findContours(frame_sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for item in contours:
        if cv2.contourArea(item) > 350:
            x,y,w,h = cv2.boundingRect(item)
            croped = frame[y:y+h, x:x+w]
            resized = cv2.resize(croped,dsize=(64,64))
            # now predict
            k = predict(resized)
            img = cv2.rectangle(framecolor,(x,y),(x+w,y+h), color=(0,255,0),thickness=3)
            if k == 1:
                cv2.putText(img,'human',org=(x,y),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=1,color=(255,0,0),bottomLeftOrigin=False)

    cv2.imshow(winname='video', mat=img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyWindow('video')
