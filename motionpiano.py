import time, cv2, math, pygame.midi
import numpy as np

NOTES = [ 60, 62, 64, 65, 67, 69, 71, 72 ]
CONSTANT_BACKGROUND = True
WIDTH = 500
NOTE_HEIGHT = 0.25
RESET_TIME = 5
SAVE_CHECK_TIME = 1

COMPARISON_VALUE = 128

savedFrame = None
savedTime = None
lastCheckTime = None

numKeys = len(NOTES)
playing = numKeys * [False]

pygame.midi.init()
player = pygame.midi.Output(0)
player.set_instrument(0)

video = cv2.VideoCapture(0)
time.sleep(2)
comparisonFrame = None

def compare(a,b):
    return cv2.threshold(cv2.absdiff(a, b), 25, COMPARISON_VALUE, cv2.THRESH_BINARY)[1]
    
while True:
    ok, frame = video.read()
    if not ok:
        time.sleep(0.05)
        continue
    if comparisonFrame is None:
        aspect = frame.shape[1] / frame.shape[0]
        height = math.floor(WIDTH / aspect)
        keys = np.zeros((height,WIDTH),np.uint8)
        blankOverlay = np.zeros((height,WIDTH,3),np.uint8)
        rects = []

        for i in range(numKeys):
            x0 = math.floor(WIDTH*i/numKeys)
            x1 = math.floor(WIDTH*(i+1)/numKeys)-1

            r = [[x0,0],[x1,math.floor(NOTE_HEIGHT*height)]]
            rects.append(r)
            cv2.rectangle(keys, r[0], r[1], color=1+i, thickness=cv2.FILLED)

    frame = cv2.flip(cv2.resize(frame, (WIDTH,height)), 1)
    blurred = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)

    if CONSTANT_BACKGROUND:
        t = time.time()
        save = False
        if savedFrame is None:
            save = True
            lastCheckTime = t
        else:
            if t >= lastCheckTime + SAVE_CHECK_TIME:
                if COMPARISON_VALUE in compare(savedFrame, blurred):
                    save = True
                lastCheckTime = t
            if t >= savedTime + RESET_TIME:
                print("resetting")
                comparisonFrame = blurred
                save = True
        if save:
            savedFrame = blurred
            savedTime = t
            
    if comparisonFrame is None:
        comparisonFrame = blurred
        continue
        
    delta = compare(comparisonFrame, blurred)
    
    sum = keys+delta
    
    overlay = blankOverlay.copy()
    for i in range(numKeys):
        if 1+i+COMPARISON_VALUE in sum:
            cv2.rectangle(overlay, rects[i][0], rects[i][1], color=(255,255,255), thickness=cv2.FILLED)
            if not playing[i]:
                player.note_on(NOTES[i],127)
                playing[i] = True
        else:
            if playing[i]:
                player.note_off(NOTES[i],127)
                playing[i] = False
        cv2.rectangle(overlay, rects[i][0], rects[i][1], color=(0,255,0))
            
    cv2.imshow("MotionPiano", cv2.addWeighted(frame, 1, overlay, 0.25, 1.0))
    if (cv2.waitKey(1) & 0xFF) == 27:
        break
    if not cv2.getWindowProperty("MotionPiano", cv2.WND_PROP_VISIBLE):
        break

    if not CONSTANT_BACKGROUND:
        comparisonFrame = blurred

video.release()
cv2.destroyAllWindows()
player.close()       