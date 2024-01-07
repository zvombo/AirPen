import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm
import math

brushThickness = 5
eraserThickness = 100

drawColor = (255, 0, 255)

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0
smooth_factor = 0.5  # Smoothing factor for position averaging
prev_x, prev_y = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        length = math.hypot(x2 - x1, y2 - y1)

        if length < 50:
            #print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # Smooth the position
            x1_smoothed = int((1 - smooth_factor) * xp + smooth_factor * x1)
            y1_smoothed = int((1 - smooth_factor) * yp + smooth_factor * y1)

            cv2.line(img, (xp, yp), (x1_smoothed, y1_smoothed), drawColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1_smoothed, y1_smoothed), drawColor, brushThickness)

            xp, yp = x1_smoothed, y1_smoothed
        else:
            xp, yp = 0, 0

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)