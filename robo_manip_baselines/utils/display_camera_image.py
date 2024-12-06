import cv2

cap = cv2.VideoCapture(12)

cv2.namedWindow("Webcam Live", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam Live", 810, 490)
cv2.moveWindow("Webcam Live", 0, 0)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        resize_frame = cv2.resize(frame, (810, 490))
        cv2.imshow("Webcam Live", resize_frame)

        if cv2.waitKey(1) & 0xFF == ord("n"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
