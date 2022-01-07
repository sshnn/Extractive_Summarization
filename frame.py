from wand.image import Image as Img
import os #dosyalama
import cv2



if not os.path.exists('image_frames'):
    os.makedirs('image_frames')


test_vid = cv2.VideoCapture('testvideo.mp4')


index = 0
while test_vid.isOpened():
    ret,frame = test_vid.read()
    if not ret:
        break


    name = './image_frames/frame' + str(index) + '.png'

    print ('Extracting frames...' + name)
    cv2.imwrite(name, frame)
    index = index + 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

test_vid.release()
cv2.destroyAllWindows()  