import cv2
from ultralytics import YOLO
import time
import sqlite3
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
from threading import Thread, Lock
import queue
import numpy as np
cv2.cuda.setDevice(0)

# Конфигурация разрешения
TARGET_WIDTH = 854
TARGET_HEIGHT = 480  # 480p resolution

# Инициализация модели
model = YOLO("yolov11n.pt").cuda()
RTSP_URL = "rtsp://user:L1k7zmWj@192.168.4.65:554/Streaming/Channels/101"

# Блокировки
db_lock = Lock()
frame_lock = Lock()

def init_db():
    with db_lock:
        conn = sqlite3.connect('detection_log.db')
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_type TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            frame_time REAL NOT NULL,
            object_id INTEGER NOT NULL
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS object_history (
            object_id INTEGER PRIMARY KEY,
            last_seen REAL NOT NULL
        )
        ''')
        conn.commit()
        conn.close()

def process_frame(frame, model, frame_time):
    with frame_lock:
        # Ресайз до 480p
        small_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # Уменьшаем качество JPEG для уменьшения нагрузки
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        _, buffer = cv2.imencode('.jpg', small_frame, encode_param)
        decoded_frame = cv2.imdecode(buffer, 1)
        
        # Трекинг объектов с пониженным разрешением
        results = model.track(
            decoded_frame, 
            persist=True, 
            verbose=False, 
            tracker="bytetrack.yaml",
            imgsz=(TARGET_HEIGHT, TARGET_WIDTH)  # Указываем размер для обработки
        )
        
        # Визуализация
        if results[0].boxes.id is not None:
            annotated_frame = results[0].plot()
        else:
            annotated_frame = decoded_frame.copy()
        
        return annotated_frame, results

def get_video_capture(rtsp_url):
    # Параметры для улучшения стабильности RTSP
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Минимизируем буфер
    cap.set(cv2.CAP_PROP_FPS, 15)        # Ограничиваем FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
    return cap

def safe_frame_read(cap):
    for _ in range(3):  # Максимум 3 попытки
        ret, frame = cap.read()
        if ret:
            return ret, frame
        # Сброс буфера при ошибке
        cap.release()
        cap = get_video_capture(RTSP_URL)
        time.sleep(0.1)
    return False, None

def video_processing_thread(q):
    init_db()
    cap = None
    last_reconnect = time.time()
    
    try:
        while True:
            # Переподключение при необходимости
            if cap is None or not cap.isOpened():
                cap = get_video_capture(RTSP_URL)
                time.sleep(1)  # Пауза после подключения
                continue
                
            # Безопасное чтение кадра
            ret, frame = safe_frame_read(cap)
            if not ret:
                print("Ошибка чтения кадра, переподключение...")
                cap.release()
                cap = None
                continue
                
            # Обработка с пониженным разрешением
            try:
                processed_frame = cv2.resize(frame, (640, 480))
                results = model.track(processed_frame, persist=True, verbose=False)
                q.put(('frame', processed_frame))
            except Exception as e:
                print(f"Ошибка обработки: {e}")
                
            time.sleep(0.05)  # Контроль нагрузки
            
    except Exception as e:
        print(f"Критическая ошибка: {e}")
    finally:
        if cap is not None:
            cap.release()

# Streamlit интерфейс
def main():
    st.title("RTSP Мониторинг с обработкой ошибок")
    video_placeholder = st.empty()
    
    cap = get_video_capture(RTSP_URL)
    last_process_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            if current_time - last_process_time < 0.1:  # 10 FPS
                time.sleep(0.01)
                continue
                
            ret, frame = cap.read()
            if not ret:
                st.warning("Ошибка получения кадра, переподключение...")
                cap.release()
                cap = get_video_capture(RTSP_URL)
                time.sleep(1)
                continue
                
            # Простая обработка
            frame = cv2.resize(frame, (640, 480))
            video_placeholder.image(frame, channels="BGR", use_container_width=True)
            
            last_process_time = current_time
            
    except Exception as e:
        st.error(f"Ошибка: {e}")
    finally:
        cap.release()

if __name__ == "__main__":
    main()