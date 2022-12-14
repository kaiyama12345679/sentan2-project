import mediapipe as mp
import numpy as np
import cv2
import time
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", help="input file name")
parser.add_argument("output", help="output file name")
args = parser.parse_args()

cap = cv2.VideoCapture(args.input)
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

tracks = []
label = True
while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.resize(img, (256, 256))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    points = {}
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                points[str(id) + "_x"] = lm.x
                points[str(id) + "_y"] = lm.y
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            if len(handLms.landmark) == 21:
                if label:
                    tracks.append(points.keys())
                tracks.append(points.values())
                label = False
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Hand Tracking", img)
    cv2.waitKey(1)

f = open(args.output, 'w', newline='')

writer = csv.writer(f)
writer.writerows(tracks)
f.close()
print("finished")
