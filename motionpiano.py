import time, cv2, math, pygame.midi
import numpy as np

notes = [ 60, 62, 64, 65, 67, 69, 71, 72 ]
constantBack = True
width = 500
noteHeight = 0.25    

numKeys = len(notes)
playing = numKeys * [False]

pygame.midi.init()
player = pygame.midi.Output(0)
player.set_instrument(0)

video = cv2.VideoCapture(0)
time.sleep(2)
comparisonFrame = None

while True:
    ok, frame = video.read()
    if not ok:
        time.sleep(0.05)
        continue
    if comparisonFrame is None:
        aspect = frame.shape[1] / frame.shape[0]
        height = math.floor(width / aspect)
        keys = np.zeros((height,width),np.uint8)
        blankOverlay = np.zeros((height,width,3),np.uint8)
        rects = []

        for i in range(numKeys):
            x0 = math.floor(width*i/numKeys)
            x1 = math.floor(width*(i+1)/numKeys)-1

            r = [[x0,0],[x1,math.floor(noteHeight*height)]]
            rects.append(r)
            cv2.rectangle(keys, r[0], r[1], color=1+2*i, thickness=cv2.FILLED)
        
    frame = cv2.flip(cv2.resize(frame, (width,height)), 1)
    blurred = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    if comparisonFrame is None:
        comparisonFrame = blurred
        continue
    delta = cv2.dilate(cv2.threshold(cv2.absdiff(comparisonFrame, blurred), 25, 1, cv2.THRESH_BINARY)[1], None, iterations=2)
    
    sum = keys+delta
    
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
            
    cv2.imshow("MotionPiano", cv2.addWeighted(frame, 1, overlay, 0.25, 1.0))
    if (cv2.waitKey(1) & 0xFF) == 27:
        break
    if not cv2.getWindowProperty("MotionPiano", cv2.WND_PROP_VISIBLE):
        break

    if not constantBack:
        comparisonFrame = blurred

video.release()
cv2.destroyAllWindows()
player.close()       