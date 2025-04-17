import cv2
from ultralytics import YOLO
import time
import sqlite3
from datetime import datetime
import streamlit as st
from threading import Thread, Lock
import queue
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Конфигурация разрешения
TARGET_WIDTH = 800
TARGET_HEIGHT = 480  # 480p resolution

# Инициализация модели
model = YOLO("yolov8n.pt")  # Убедитесь что файл модели существует
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

def log_detection(object_type, timestamp, frame_time, object_id):
    with db_lock:
        conn = sqlite3.connect('detection_log.db')
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO detections (object_type, timestamp, frame_time, object_id)
        VALUES (?, ?, ?, ?)
        ''', (object_type, timestamp, frame_time, object_id))
        conn.commit()
        conn.close()

def process_frame(frame, model, frame_time):
    with frame_lock:
        try:
            # Ресайз до 480p
            small_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
            
            # Конвертация цветового пространства (важно для YOLO)
            frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Трекинг объектов
            results = model.track(
                frame_rgb,  # Используем RGB
                persist=True, 
                verbose=False, 
                tracker="bytetrack.yaml",
                imgsz=(TARGET_HEIGHT, TARGET_WIDTH)
            )
            # Логирование обнаруженных объектов
            if results[0].boxes.id is not None:
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                for obj in results[0].boxes:
                    label = model.names[int(obj.cls)]
                    if label in ['car', 'truck', 'person'] and obj.id is not None:
                        log_detection(
                            label,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            frame_time,
                            int(obj.id)
                        )
            else:
                annotated_frame = small_frame.copy()
            
            return annotated_frame, results
        except Exception as e:
            print(f"Ошибка в process_frame: {e}")
            return small_frame, None

def get_video_capture(rtsp_url):
    # Настройки для стабильного RTSP соединения
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Уменьшите буфер для снижения задержки
    cap.set(cv2.CAP_PROP_FPS, 15)        # Установите FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    return cap

def video_processing_thread(q):
    init_db()
    cap = None
    
    while True:
        try:
            if cap is None or not cap.isOpened():
                cap = get_video_capture(RTSP_URL)
                time.sleep(1)
                continue
                
            ret, frame = cap.read()
            if not ret:
                print("Ошибка чтения кадра, переподключение...")
                cap.release()
                cap = None
                continue
                
            # Обработка кадра
            processed_frame, results = process_frame(frame, model, time.time())
            
            # Отладочный вывод
            if results and results[0].boxes.id is not None:
                print(f"Обнаружено объектов: {len(results[0].boxes)}")
                
            q.put(('frame', processed_frame))
            
        except Exception as e:
            print(f"Ошибка в video_processing_thread: {e}")
            if cap is not None:
                cap.release()
                cap = None
            time.sleep(1)

def main():
    st.title("RTSP Мониторинг с детекцией объектов")
    
    # Инициализация очереди и потока
    if 'frame_queue' not in st.session_state:
        st.session_state.frame_queue = queue.Queue(maxsize=1)
        st.session_state.thread = Thread(
            target=video_processing_thread,
            args=(st.session_state.frame_queue,),
            daemon=True
        )
        st.session_state.thread.start()
    
    # Интерфейс
    video_placeholder = st.empty()
    status_text = st.empty()
    
    while True:
        try:
            item_type, frame = st.session_state.frame_queue.get_nowait()
            if item_type == 'frame':
                video_placeholder.image(frame, channels="BGR", use_container_width=True)
                status_text.success("Детекция активна")
        except queue.Empty:
            status_text.warning("Ожидание кадров...")
            time.sleep(0.1)

if __name__ == "__main__":
    main()