import time, cv2, math, pygame.midi
import numpy as np

NOTES = [ 60, 62, 64, 65, 67, 69, 71, 72, 74 ] # , 76, 77, 79 ]
CONSTANT_BACKGROUND = True
WIDTH = 500
KERNEL_SIZE = 0.042
KEY_HEIGHT = 0.25
RESET_TIME = 5
SAVE_CHECK_TIME = 1
THRESHOLD = 25
WINDOW_NAME = "MotionPiano"
NOTE_VELOCITY = 127

COMPARISON_VALUE = 128

savedFrame = None
savedTime = None
lastCheckTime = None

numKeys = len(NOTES)
playing = numKeys * [False]
kernelSize = 2*int(KERNEL_SIZE*WIDTH/2)+1

pygame.midi.init()

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

id = None
for i in range(pygame.midi.get_count()):
    info = pygame.midi.get_device_info(i)
    if info[3]:
        if id is None:
            id = i
        if b"timidity" in info[1].lower():
            id = i
            break

player = pygame.midi.Output(id)
player.set_instrument(0)

video = cv2.VideoCapture(0)
#video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
#video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
comparisonFrame = None

def compare(a,b):
    return cv2.threshold(cv2.absdiff(a, b), THRESHOLD, COMPARISON_VALUE, cv2.THRESH_BINARY)[1]
    
while True:
    ok, frame = video.read()
    if not ok:
        time.sleep(0.05)
        continue
    if comparisonFrame is None:
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        cv2.resizeWindow(WINDOW_NAME, frameWidth, frameHeight)
        aspect = frameWidth / frameHeight
        scaledWidth = WIDTH
        scaledHeight = int(WIDTH / aspect)
        
        blankOverlay = np.zeros((frameHeight,frameWidth,3),dtype=np.uint8)

        frameRects = []
        scaledRects = []

        for i in range(numKeys):
            x0 = scaledWidth*i//numKeys
            x1 = scaledWidth*(i+1)//numKeys-1

            r = [(x0,0),(x1,int(KEY_HEIGHT*scaledHeight))]
            scaledRects.append(r)

            x0 = frameWidth*i//numKeys
            x1 = frameWidth*(i+1)//numKeys-1

            r = [(x0,0),(x1,int(KEY_HEIGHT*frameHeight))]
            frameRects.append(r)
            
        keysTopLeftScaled = (min(r[0][0] for r in scaledRects),min(r[0][1] for r in scaledRects))
        keysBottomRightScaled = (max(r[1][0] for r in scaledRects),max(r[1][1] for r in scaledRects))
        keysWidthScaled = keysBottomRightScaled[0]-keysTopLeftScaled[0]
        keysHeightScaled = keysBottomRightScaled[1]-keysTopLeftScaled[1]
        keysTopLeftFrame = (min(r[0][0] for r in frameRects),min(r[0][1] for r in frameRects))
        keysBottomRightFrame = (max(r[1][0] for r in frameRects),max(r[1][1] for r in frameRects))
        keys = np.zeros((keysHeightScaled,keysWidthScaled),dtype=np.uint8)
        
        def adjustToKeys(xy):
            return (xy[0]-keysTopLeftScaled[0],xy[1]-keysTopLeftScaled[1])
            
        for i in range(numKeys):
            r = scaledRects[i]
            cv2.rectangle(keys, adjustToKeys(r[0]), adjustToKeys(r[1]), i+1, cv2.FILLED)

    frame = cv2.flip(frame, 1)
    keysFrame = frame[keysTopLeftFrame[1]:keysBottomRightFrame[1], keysTopLeftFrame[0]:keysBottomRightFrame[0]]
    keysFrame = cv2.resize(keysFrame, (keysWidthScaled,keysHeightScaled))
    blurred = cv2.GaussianBlur(cv2.cvtColor(keysFrame, cv2.COLOR_BGR2GRAY), (kernelSize, kernelSize), 0)

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
            cv2.rectangle(overlay, frameRects[i][0], frameRects[i][1], (255,255,255), cv2.FILLED)
            if not playing[i]:
                player.note_on(NOTES[i],NOTE_VELOCITY152501046)
                playing[i] = True
        else:
            if playing[i]:
                player.note_off(NOTES[i],NOTE_VELOCITY)
                playing[i] = False
        cv2.rectangle(overlay, frameRects[i][0], frameRects[i][1], (0,255,0), 1)
            
    cv2.imshow(WINDOW_NAME, cv2.addWeighted(frame, 1, overlay, 0.25, 1.0))
    if (cv2.waitKey(1) & 0xFF) == 27:
        break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) <= 0:
        break

    if not CONSTANT_BACKGROUND:
        comparisonFrame = blurred

video.release()
cv2.destroyAllWindows()
player.close()       
