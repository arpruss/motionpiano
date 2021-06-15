from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
import math
import pygame.midi
import time

notes = [ 60, 62, 64, 65, 67, 69, 71, 72 ]
playing = len(notes) * [False]

pygame.midi.init()
player = pygame.midi.Output(0)
player.set_instrument(0)

constantBack = True

width = 500
height = 500 * 3//4
numKeys = len(notes)
noteHeight = 0.25

keys = np.zeros((height,width),np.uint8)
blankOverlay = np.zeros((height,width,3),np.uint8)
rects = []

for i in range(numKeys):
    x0 = math.floor(width*i/numKeys)
    x1 = math.floor(width*(i+1)/numKeys)-1

    r = [[x0,0],[x1,math.floor(noteHeight*height)]]
    rects.append(r)
    cv2.rectangle(keys, r[0], r[1], color=1+2*i, thickness=cv2.FILLED)
    
vs = VideoStream(src=0).start()
time.sleep(2.0)
prevFrame = None

while True:
    frame = vs.read()
    if frame is None:
        break
    frame = cv2.flip(imutils.resize(frame, width=width, height=height), 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if prevFrame is None:
        prevFrame = gray
        continue
    frameDelta = cv2.absdiff(prevFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 1, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    sum = keys+thresh
    
    overlay = blankOverlay.copy()
    for i in range(numKeys):
        if 1+2*i+1 in sum:
            cv2.rectangle(overlay, rects[i][0], rects[i][1], color=(255,255,255), thickness=cv2.FILLED)
            if not playing[i]:
                player.note_on(notes[i],127)
                playing[i] = True
        else:
            if playing[i]:
                player.note_off(notes[i],127)
                playing[i] = False
        cv2.rectangle(overlay, rects[i][0], rects[i][1], color=(0,255,0))
            
    frame = cv2.addWeighted(frame, 1, overlay, 0.25, 1.0)
    
    cv2.imshow("Feed", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break
    if not cv2.getWindowProperty('Feed', cv2.WND_PROP_VISIBLE):
        break

    if not constantBack:
        prevFrame = gray

vs.stop()
cv2.destroyAllWindows()
player.close()       