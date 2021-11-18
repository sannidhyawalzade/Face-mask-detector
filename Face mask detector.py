import numpy as np
import cv2


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
origin = (60, 60)
green = (0, 255, 0)
red = (0, 0, 255)
orange = (0, 123, 255)
thickness = 4
font_scale = 1.5

cam = cv2.VideoCapture(0)

while True:
    success, frame = cam.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    (thresh, bw_img) = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    # this is added to fix the issue that occurs when a light grey or white mask is encountered.

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    faces_bw = face_cascade.detectMultiScale(bw_img, 1.1, 4)

    if (len(faces) == 0 and len(faces_bw) == 0):
        cv2.putText(frame, "Finding Smiles ???? ", origin, font,
                    font_scale, orange, thickness, cv2.LINE_AA)

    elif (len(faces) == 0 and len(faces_bw) == 1):
        cv2.putText(frame, "Mask on ! Appreciated !!!", origin, font,
                    font_scale, green, thickness, cv2.LINE_AA)

    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            lips = mouth_cascade.detectMultiScale(gray, 1.5, 5)

        if(len(lips) == 0):
            cv2.putText(frame, "Mask on!😷😷😷 Appreciated", origin,
                        font, font_scale, green, thickness, cv2.LINE_AA)
        else:
            for (mx, my, mw, mh) in lips:
                if(y < my < y + h):
                    cv2.putText(frame, "Alert !!! Put on a Mask.", origin, font, font_scale,
                                red, thickness, cv2.LINE_AA)

                    break

    cv2.imshow('Mask Detector', frame)
    k = cv2.waitKey(30)
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
