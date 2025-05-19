import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt


vid_path = "C:\\Users\\Admin\\Desktop\\MyWork\\test5.mp4"
cap = cv2.VideoCapture(vid_path)
backSub = cv2.createBackgroundSubtractorMOG2()
model = YOLO('yolo11n.pt')

if not cap.isOpened():
    print("Error opening video file")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Вычитание фона
        fg_mask = backSub.apply(frame)

        # Поиск контура на обработанной маске
        retval, mask_thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
        
        contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        min_contour_area = 1000
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        if large_contours is not None:
            results = model(frame)
            annotated_frame = results[0].plot()

        frame_out = frame.copy()
        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            frame_out = cv2.rectangle(frame_out, (x, y), (x+w, y+h), (0, 0, 200), 3)

        # Отображение результатов
        cv2.imshow('Gray_Frame', mask_eroded)
        cv2.imshow('Detected_Movement', frame_out)
        cv2.imshow('Frame_Final', annotated_frame)
        
        # Ожидание для корректного отображения
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()