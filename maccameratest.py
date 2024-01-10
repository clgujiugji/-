import cv2 as cv

cap = cv.VideoCapture(0)

while True:
    success, frame = cap.read()

    if success is False:
        print('read video false!')
        exit(0)

    cv.imshow('video', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
