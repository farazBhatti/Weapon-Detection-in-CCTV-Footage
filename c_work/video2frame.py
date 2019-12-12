import numpy as np
import cv2

path = './videos/v_1.mp4'
save_dir = './imgs/'
cap = cv2.VideoCapture(path)


# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        #frame = cv2.flip(frame,0)

        # write the flipped frame
        name = save_dir + 'image_' + str(i) + '.png'
        cv2.imwrite(name,frame)
        i = i + 1

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()

cv2.destroyAllWindows()