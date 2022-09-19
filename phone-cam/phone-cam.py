import cv2

# change your IP
cap = cv2.VideoCapture('http://192.168.0.101:8080/video')


while(cap.isOpened()):

    ret, frame = cap.read()

    try:
        cv2.imshow('temp', cv2.resize(frame, (600,400)))

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except cv2.error:
        print("Stream ended...")
        break

        
cap.release()
cv2.destroyAllWindows()
