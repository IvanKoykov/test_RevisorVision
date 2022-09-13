import os
import cv2

def extract():
    if not os.path.exists("image"):
        os.mkdir("image")
    cam=cv2.VideoCapture("video.mp4")
    currentframe=0
    xmin,ymin = 115, 210
    xmax, ymax = 350, 445
    while(True):
        ret,frame=cam.read()
        if ret:
            name='image/image_'+str(currentframe)+'.jpg'
            if currentframe>0 and currentframe % 15==0:
                frame=frame[ymin:ymax, xmin:xmax]
                frame=cv2.resize(frame,(116,116))
                cv2.imwrite(name,frame)
                currentframe+=1
            else:
                currentframe+=1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    extract()