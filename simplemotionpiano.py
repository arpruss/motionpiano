import time, cv2
import numpy as np
import rtmidi

NOTES = [ 60, 62, 64, 65, 67, 69, 71, 72, 74 ] # , 76, 77, 79 ]
NOTE_VELOCITY = 127
WINDOW_NAME = "MotionPiano"
KEY_HEIGHT = 0.25

RECOGNIZER_WIDTH = 500
KERNEL_SIZE = 0.042
RESET_TIME = 5
SAVE_CHECK_TIME = 1
THRESHOLD = 25
COMPARISON_VALUE = 128

numKeys = len(NOTES)
playing = numKeys * [False]

midiout = rtmidi.MidiOut()
assert(midiout.get_ports())
midiout.open_port(0)

video = cv2.VideoCapture(0)
frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

if RECOGNIZER_WIDTH >= frameWidth:
    scaledWidth = frameWidth
    scaledHeight = frameHeight
else:
    aspect = frameWidth / frameHeight
    scaledWidth = RECOGNIZER_WIDTH
    scaledHeight = int(RECOGNIZER_WIDTH / aspect)
    
kernelSize = 2*int(KERNEL_SIZE*scaledWidth/2)+1

blankOverlay = np.zeros((frameHeight,frameWidth,3),dtype=np.uint8)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(WINDOW_NAME, frameWidth, frameHeight)

scaledRects = []
frameRects = []

for i in range(numKeys):
    x0 = scaledWidth*i//numKeys
    x1 = scaledWidth*(i+1)//numKeys-1

    r = [(x0,0),(x1,int(KEY_HEIGHT*scaledHeight))]
    scaledRects.append(r)

    x0 = frameWidth*i//numKeys
    x1 = frameWidth*(i+1)//numKeys-1

    r = [(x0,0),(x1,int(KEY_HEIGHT*frameHeight))]
    frameRects.append(r)
    
keysTopLeftFrame = (min(r[0][0] for r in frameRects),min(r[0][1] for r in frameRects))
keysBottomRightFrame = (max(r[1][0] for r in frameRects),max(r[1][1] for r in frameRects))

keysTopLeftScaled = (min(r[0][0] for r in scaledRects),min(r[0][1] for r in scaledRects))
keysBottomRightScaled = (max(r[1][0] for r in scaledRects),max(r[1][1] for r in scaledRects))
keysWidthScaled = keysBottomRightScaled[0]-keysTopLeftScaled[0]
keysHeightScaled = keysBottomRightScaled[1]-keysTopLeftScaled[1]

keys = np.zeros((keysHeightScaled,keysWidthScaled),dtype=np.uint8)

def adjustToKeys(xy):
    return (xy[0]-keysTopLeftScaled[0],xy[1]-keysTopLeftScaled[1])
    
for i in range(numKeys):
    r = scaledRects[i]
    cv2.rectangle(keys, adjustToKeys(r[0]), adjustToKeys(r[1]), i+1, cv2.FILLED)

comparisonFrame = None
savedFrame = None
savedTime = 0
lastCheckTime = 0

def compare(a,b):
    return cv2.threshold(cv2.absdiff(a, b), THRESHOLD, COMPARISON_VALUE, cv2.THRESH_BINARY)[1]
    
while True:
    ok, frame = video.read()
    if not ok:
        time.sleep(0.05)
        continue
    frame = cv2.flip(frame, 1)

    keysFrame = frame[keysTopLeftFrame[1]:keysBottomRightFrame[1], keysTopLeftFrame[0]:keysBottomRightFrame[0]]
    if scaledWidth != frameWidth:
        keysFrame = cv2.resize(keysFrame, (keysWidthScaled,keysHeightScaled))
    keysFrame = cv2.cvtColor(keysFrame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(keysFrame, (kernelSize, kernelSize), 0)

    t = time.time()
    save = False
    if savedFrame is None:
        save = True
        lastCheckTime = t
    else:
        if t >= lastCheckTime + SAVE_CHECK_TIME:
            if COMPARISON_VALUE in compare(savedFrame, blurred):
                print("saving")
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
            r = frameRects[i]
            cv2.rectangle(overlay, r[0], r[1], (255,255,255), cv2.FILLED)
            if not playing[i]:
                midiout.send_message([0x90, NOTES[i], NOTE_VELOCITY])
                playing[i] = True
        else:
            if playing[i]:
                midiout.send_message([0x80, NOTES[i], 0])
                playing[i] = False
        cv2.rectangle(overlay, r[0], r[1], (0,255,0), 1)

    cv2.imshow(WINDOW_NAME, cv2.addWeighted(frame, 1, overlay, 0.25, 1.0))
    
    if (cv2.waitKey(1) & 0xFF) == 27 or cv2.getWindowProperty(WINDOW_NAME, 0) == -1:
        break

video.release()
cv2.destroyAllWindows()
del midiout 
