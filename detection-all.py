import cv2
from ultralytics import YOLO
import time
import sqlite3
from datetime import datetime
import streamlit as st
from threading import Thread, Lock
import queue
import os
import torch

# Конфигурация
TARGET_WIDTH = 864
TARGET_HEIGHT = 480
DB_PATH = 'detection_log.db'

# Инициализация блокировки для БД
db_lock = Lock()

# Настройка torch для избежания проблем с CUDA
torch.backends.cudnn.enabled = False

# Модель YOLO (принудительно на CPU если есть проблемы)
try:
    model = YOLO("yolov8n.pt").to('cpu')  # Можно заменить на 'cuda:0' если GPU работает
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = YOLO("yolov8n.pt")  # Без указания устройства

# RTSP URLs
RTSP_URLS = {
    "Camera_1": "rtsp://user:L1k7zmWj@192.168.14.92:554/Streaming/Channels/101",
    "Camera_2": "rtsp://user:0704RRrr@192.168.4.65:554/Streaming/Channels/101"
}


def init_db():
    conn = None
    try:
        with db_lock:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            # Проверяем существование таблицы
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detections'")
            if not cursor.fetchone():
                cursor.execute('''
                CREATE TABLE detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_type TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    frame_time REAL NOT NULL,
                    object_id INTEGER NOT NULL,
                    camera TEXT NOT NULL
                )
                ''')
                print("Таблица detections создана")

            # Проверяем наличие колонки camera
            cursor.execute("PRAGMA table_info(detections)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'camera' not in columns:
                cursor.execute("ALTER TABLE detections ADD COLUMN camera TEXT NOT NULL DEFAULT 'unknown'")
                print("Добавлена колонка camera")

            conn.commit()
            print("База данных успешно инициализирована")
            return True
    except Exception as e:
        print(f"Ошибка инициализации БД: {e}")
        return False
    finally:
        if conn:
            conn.close()


def log_detection(object_type, timestamp, frame_time, object_id, camera):
    max_retries = 3
    retry_delay = 0.1

    # Проверка входных данных
    if None in (object_type, timestamp, frame_time, object_id, camera):
        print("Ошибка: Обнаружены None значения, запись отменена")
        return False

    for attempt in range(max_retries):
        conn = None
        try:
            with db_lock:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()

                cursor.execute('''
                INSERT INTO detections (object_type, timestamp, frame_time, object_id, camera)
                VALUES (?, ?, ?, ?, ?)
                ''', (str(object_type), str(timestamp), float(frame_time), int(object_id), str(camera)))

                conn.commit()
                print(f"Запись в БД успешна: {object_type} (ID: {object_id}) с камеры {camera}")
                return True

        except sqlite3.Error as e:
            print(f"Ошибка записи в БД (попытка {attempt + 1}): {e}")
            time.sleep(retry_delay)
        except Exception as e:
            print(f"Неожиданная ошибка при записи в БД: {e}")
            break
        finally:
            if conn:
                conn.close()

    print("Не удалось записать данные в БД после нескольких попыток")
    return False


def verify_db_entries():
    """Функция для проверки записанных данных"""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM detections")
        count = cursor.fetchone()[0]
        print(f"Всего записей в БД: {count}")

        if count > 0:
            cursor.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 5")
            print("Последние 5 записей:")
            for row in cursor.fetchall():
                print(row)
        return count > 0
    except Exception as e:
        print(f"Ошибка проверки БД: {e}")
        return False
    finally:
        if conn:
            conn.close()


def process_frame(frame, model, frame_time, camera_name, rotate_180=False):
    try:
        # Переворот кадра на 180 градусов для первой камеры
        if rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Ресайз до 480p
        small_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Трекинг объектов с обработкой ошибок NMS
        try:
            with torch.no_grad():
                results = model.track(
                    frame_rgb,
                    persist=True,
                    verbose=False,
                    tracker="bytetrack.yaml",
                    imgsz=(TARGET_HEIGHT, TARGET_WIDTH),
                    device='cpu'  # Принудительно используем CPU для избежания ошибок NMS
                )
        except Exception as e:
            print(f"Ошибка трекинга: {e}, пробуем простой predict")
            results = model.predict(
                frame_rgb,
                imgsz=(TARGET_HEIGHT, TARGET_WIDTH),
                device='cpu'
            )

        # Логирование обнаруженных объектов
        if results and hasattr(results[0], 'boxes') and results[0].boxes.id is not None:
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            for obj in results[0].boxes:
                label = model.names[int(obj.cls)]
                obj_id = int(obj.id) if obj.id is not None else 0
                if label in ['car', 'truck', 'person']:
                    log_detection(
                        label,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        frame_time,
                        obj_id,
                        camera_name
                    )
        else:
            annotated_frame = small_frame.copy()

        return annotated_frame, results
    except Exception as e:
        print(f"Критическая ошибка в process_frame: {e}")
        return small_frame, None


def get_video_capture(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Не удалось открыть соединение с {rtsp_url}")
        return None

    # Настройки для уменьшения задержки
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    return cap


def video_processing_thread(rtsp_url, queue, camera_name, rotate_180=False):
    print(f"Запуск потока обработки для {camera_name}")
    init_db()
    cap = None
    last_check_time = time.time()

    while True:
        try:
            current_time = time.time()

            # Проверяем соединение каждые 10 секунд
            if current_time - last_check_time > 10:
                if not verify_db_entries():
                    print("Проблема с БД, переинициализация...")
                    init_db()
                last_check_time = current_time

            if cap is None or not cap.isOpened():
                cap = get_video_capture(rtsp_url)
                if cap is None:
                    time.sleep(1)
                    continue

            ret, frame = cap.read()
            if not ret:
                print(f"Ошибка чтения кадра из {rtsp_url}, переподключение...")
                cap.release()
                cap = None
                time.sleep(1)
                continue

            # Обработка кадра
            frame_time = time.time()
            processed_frame, results = process_frame(frame, model, frame_time, camera_name, rotate_180)

            # Отладочный вывод
            if results and hasattr(results[0], 'boxes') and results[0].boxes.id is not None:
                print(f"Камера {camera_name}: Обнаружено {len(results[0].boxes)} объектов")

            # Помещаем кадр в очередь
            if not queue.full():
                queue.put(('frame', processed_frame))
            else:
                # Если очередь полна, извлекаем старый кадр перед добавлением нового
                try:
                    queue.get_nowait()
                except queue.Empty:
                    pass
                queue.put(('frame', processed_frame))

            time.sleep(0.01)  # Небольшая задержка для снижения нагрузки

        except Exception as e:
            print(f"Критическая ошибка в video_processing_thread ({camera_name}): {e}")
            if cap is not None:
                cap.release()
                cap = None
            time.sleep(1)


def main():
    st.set_page_config(layout="wide")
    st.title("Мониторинг с детекцией объектов (2 камеры)")

    # Проверяем и создаем БД при запуске
    if not os.path.exists(DB_PATH):
        print("База данных не найдена, создаем новую...")
        init_db()

    # Проверяем структуру БД
    verify_db_entries()

    # Инициализация очередей и потоков
    if 'frame_queue_1' not in st.session_state:
        st.session_state.frame_queue_1 = queue.Queue(maxsize=1)
        st.session_state.thread_1 = Thread(
            target=video_processing_thread,
            args=(RTSP_URLS["Camera_1"], st.session_state.frame_queue_1, "Camera_1", True),
            daemon=True
        )
        st.session_state.thread_1.start()

    if 'frame_queue_2' not in st.session_state:
        st.session_state.frame_queue_2 = queue.Queue(maxsize=1)
        st.session_state.thread_2 = Thread(
            target=video_processing_thread,
            args=(RTSP_URLS["Camera_2"], st.session_state.frame_queue_2, "Camera_2", False),
            daemon=True
        )
        st.session_state.thread_2.start()

    # Интерфейс
    col1, col2 = st.columns(2)
    placeholder_1 = col1.empty()
    placeholder_2 = col2.empty()
    status_text_1 = col1.empty()
    status_text_2 = col2.empty()

    last_update_time = time.time()

    while True:
        try:
            current_time = time.time()

            # Получение кадров из очередей
            if not st.session_state.frame_queue_1.empty():
                item_type, frame = st.session_state.frame_queue_1.get_nowait()
                if item_type == 'frame':
                    placeholder_1.image(frame, channels="BGR", use_container_width=True)
                    status_text_1.success(f"Камера 1: Активна ({datetime.now().strftime('%H:%M:%S')})")
                    last_update_time = current_time

            if not st.session_state.frame_queue_2.empty():
                item_type, frame = st.session_state.frame_queue_2.get_nowait()
                if item_type == 'frame':
                    placeholder_2.image(frame, channels="BGR", use_container_width=True)
                    status_text_2.success(f"Камера 2: Активна ({datetime.now().strftime('%H:%M:%S')})")
                    last_update_time = current_time

            # Проверка "зависания" интерфейса
            if current_time - last_update_time > 5:
                status_text_1.warning("Камера 1: Ожидание кадров...")
                status_text_2.warning("Камера 2: Ожидание кадров...")

            time.sleep(0.1)

        except queue.Empty:
            time.sleep(0.1)
        except Exception as e:
            print(f"Ошибка в основном цикле: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()