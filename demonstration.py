import mediapipe as mp
import numpy as np
import cv2
import time
import csv
import argparse
import torch
from classification import Network

model = Network(4)

parser = argparse.ArgumentParser()
parser.add_argument("model", help="model file name")
args = parser.parse_args()

model.load_state_dict(torch.load(args.model))

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

pred = -1
while True:
    success, img = cap.read()
    if not success:
        print("error")
        break
    img = cv2.resize(img, (256, 256))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    points = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append(lm.x)
                points.append(lm.y)

                #print(id, cx, cy)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            if len(points) == 42:
                x = torch.tensor(points, dtype=torch.float64, requires_grad=False)
                x = x.float()
                y = model.forward(x)
                print(torch.argmax(y))
                pred = torch.argmax(y).item()
            else:
                pred = -1
    else:
        pred = -1
                
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    if pred == 0:
        message = "good"
    elif pred == 1:
        message = "bad"
    elif pred == 2:
        message = "open"
    elif pred == 3:
        message = "close"
    else:
        message = "unknown"
    cv2.putText(img, message, (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Hand Tracking", img)
    cv2.waitKey(1)
