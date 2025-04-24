import cv2
from ultralytics import YOLO
import time
import sqlite3
from datetime import datetime
import streamlit as st
from threading import Thread, Lock
import queue


# Конфигурация разрешения
TARGET_WIDTH = 864  # Кратно 32
TARGET_HEIGHT = 480  # Кратно 32

# Модель YOLO
model = YOLO("yolov8n.pt")  # Убедитесь что файл модели существует

# RTSP-URL для камер
RTSP_URL_1 = "rtsp://user:0704RRrr@192.168.4.65:554/Streaming/Channels/101"
RTSP_URL_2 = "rtsp://user:L1k7zmWj@192.168.14.92:554/Streaming/Channels/101"

# Блокировки и очереди
db_lock = Lock()
frame_queue_1 = queue.Queue(maxsize=1)
frame_queue_2 = queue.Queue(maxsize=1)

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
    try:
        # Ресайз до 480p
        small_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Трекинг объектов
        results = model.track(
            frame_rgb,
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
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    return cap

def video_processing_thread(rtsp_url, queue):
    init_db()
    cap = None
    
    while True:
        try:
            if cap is None or not cap.isOpened():
                cap = get_video_capture(rtsp_url)
                time.sleep(1)
                continue
                
            ret, frame = cap.read()
            if not ret:
                print(f"Ошибка чтения кадра из {rtsp_url}, переподключение...")
                cap.release()
                cap = None
                continue
                
            # Обработка кадра
            processed_frame, results = process_frame(frame, model, time.time())
            
            # Отладочный вывод
            if results and results[0].boxes.id is not None:
                print(f"Обнаружено объектов: {len(results[0].boxes)}")
                
            # Помещаем кадр в очередь
            if not queue.full():
                queue.put(('frame', processed_frame))
            
        except Exception as e:
            print(f"Ошибка в video_processing_thread ({rtsp_url}): {e}")
            if cap is not None:
                cap.release()
                cap = None
            time.sleep(1)

def main():
    st.title("Мониторинг с детекцией объектов (2 камеры)")
    
    # Инициализация очередей и потоков
    if 'frame_queue_1' not in st.session_state:
        st.session_state.frame_queue_1 = queue.Queue(maxsize=1)
        st.session_state.thread_1 = Thread(
            target=video_processing_thread,
            args=(RTSP_URL_1, st.session_state.frame_queue_1),
            daemon=True
        )
        st.session_state.thread_1.start()
    
    if 'frame_queue_2' not in st.session_state:
        st.session_state.frame_queue_2 = queue.Queue(maxsize=1)
        st.session_state.thread_2 = Thread(
            target=video_processing_thread,
            args=(RTSP_URL_2, st.session_state.frame_queue_2),
            daemon=True
        )
        st.session_state.thread_2.start()
    
    # Интерфейс
    col1, col2 = st.columns(2)
    placeholder_1 = col1.empty()
    placeholder_2 = col2.empty()
    status_text_1 = col1.empty()
    status_text_2 = col2.empty()
    
    while True:
        try:
            # Получение кадров из очередей
            if not st.session_state.frame_queue_1.empty():
                item_type, frame = st.session_state.frame_queue_1.get_nowait()
                if item_type == 'frame':
                    placeholder_1.image(frame, channels="BGR", use_container_width=True)
                    status_text_1.success("Камера 1: Детекция активна")
            
            if not st.session_state.frame_queue_2.empty():
                item_type, frame = st.session_state.frame_queue_2.get_nowait()
                if item_type == 'frame':
                    placeholder_2.image(frame, channels="BGR", use_container_width=True)
                    status_text_2.success("Камера 2: Детекция активна")
        except queue.Empty:
            status_text_1.warning("Камера 1: Ожидание кадров...")
            status_text_2.warning("Камера 2: Ожидание кадров...")
            time.sleep(0.1)

if __name__ == "__main__":
    main()