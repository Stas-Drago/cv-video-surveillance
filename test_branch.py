import cv2
cap = cv2.VideoCapture("rtsp://user:L1k7zmWj@192.168.4.65:554/Streaming/Channels/101")
ret, frame = cap.read()
print("Кадр получен:", ret)
cap.release()