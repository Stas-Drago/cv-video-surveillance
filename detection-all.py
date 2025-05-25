import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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


# Глобальный трекер

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
    model = YOLO("C:\\Users\\Admin\\Desktop\\MyWork\\yolo11n.pt").to('cpu')  # Можно заменить на 'cuda:0' если GPU работает
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = YOLO("yolov11n.pt")  # Без указания устройства

# RTSP URLs
RTSP_URLS = {
    "Camera_1": "rtsp://user:L1k7zmWj@192.168.14.92:554/Streaming/Channels/101",
    "Camera_2": "rtsp://user:0704RRrr@192.168.4.65:554/Streaming/Channels/101"
}


class UniqueObjectTracker:
    def __init__(self, max_age=10):
        self.active_objects = {}  # {obj_id: {'bbox': [x,y,w,h], 'first_seen': timestamp}}
        self.max_age = max_age  # Количество кадров для хранения "исчезнувшего" объекта
        self.next_id = 0

    def register(self, bbox):
        """Регистрация нового объекта"""
        self.active_objects[self.next_id] = {
            'bbox': bbox,
            'first_seen': datetime.now(),
            'last_seen': datetime.now()
        }
        obj_id = self.next_id
        self.next_id += 1
        return obj_id

    def update(self, detections, camera):
        """Обновление списка объектов"""
        updated = {}
        dets = [(det['bbox'], det['label']) for det in detections]
        
        # Сопоставление новых детекций с активными объектами
        for det_bbox, label in dets:
            matched = False
            for obj_id, obj_data in self.active_objects.items():
                if self.iou(det_bbox, obj_data['bbox']) > 0.3:
                    updated[obj_id] = {
                        'bbox': det_bbox,
                        'first_seen': obj_data['first_seen'],
                        'last_seen': datetime.now()
                    }
                    matched = True
                    break
            if not matched:
                new_id = self.register(det_bbox)
                updated[new_id] = {
                    'bbox': det_bbox,
                    'first_seen': datetime.now(),
                    'last_seen': datetime.now()
                }

        self.active_objects = updated
        for obj_id in list(self.active_objects.keys()):
            if obj_id not in updated:
                if (datetime.now() - self.active_objects[obj_id]['last_seen']).total_seconds() > self.max_age: # Укажите камеру динамически
                    update_exit_time(obj_id, camera)
                    del self.active_objects[obj_id]

        return self.active_objects

    def iou(self, box1, box2):
        """Вычисление Intersection over Union"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1 + w1, x2 + w2)
        yB = min(y1 + h1, y2 + h2)
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        box1Area = w1 * h1
        box2Area = w2 * h2
        
        return interArea / float(box1Area + box2Area - interArea)

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
                camera TEXT NOT NULL,
                entered_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                exited_at DATETIME NULL
                )
                ''')
                print("Таблица detections создана")

            # Проверяем наличие колонок
            cursor.execute("PRAGMA table_info(detections)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'entered_at' not in columns:
                cursor.execute("ALTER TABLE detections ADD COLUMN entered_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP")
            if 'exited_at' not in columns:
                cursor.execute("ALTER TABLE detections ADD COLUMN exited_at DATETIME NULL")
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

def update_exit_time(object_id, camera):
    try:
        with db_lock:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
            UPDATE detections 
            SET exited_at = ?
            WHERE object_id = ? AND camera = ? AND exited_at IS NULL
            ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), object_id, camera))
            conn.commit()
    except Exception as e:
        print(f"Ошибка обновления времени выхода: {e}")
    finally:
        if conn:
            conn.close()


def verify_db_entries():
    """Проверяет, есть ли данные в БД"""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM detections")
        count = cursor.fetchone()[0]
        print(f"Всего записей в БД: {count}")
        return count > 0
    except sqlite3.OperationalError as e:
        print(f"Ошибка доступа к БД: {e}")
        return False
    finally:
        if conn:
            conn.close()

global_tracker = UniqueObjectTracker()
def process_frame(frame, model, frame_time, camera_name, rotate_180=False):
    try:
        # Переворот и ресайз кадра
        if rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        small_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Детекция объектов
        results = model.track(
            frame_rgb,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
            imgsz=(TARGET_HEIGHT, TARGET_WIDTH),
            device='cpu'
        )

        # Сборка детекций
        # Сборка детекций
        detections = []
        if results and hasattr(results[0], 'boxes') and results[0].boxes.id is not None:
            for obj in results[0].boxes:
                try:
                    x1, y1, x2, y2 = obj.xyxy[0].tolist()
                    bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                    if hasattr(obj, 'cls') and 0 <= int(obj.cls) < len(model.names):
                        label = model.names[int(obj.cls)]
                    else:
                        label = 'unknown'
                    detections.append({'bbox': bbox, 'label': label})
                except Exception as e:
                    print(f"Ошибка обработки объекта: {e}")
                    continue
        else:
            # Нет детекций — возвращаем пустой список
            detections = []

# Обновление глобального трекера
        tracked_objects = global_tracker.update(detections, camera_name)

        # Аннотация кадра
        if results and hasattr(results[0], 'boxes') and results[0].boxes.id is not None:
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            if tracked_objects:
                for obj_id, obj_data in tracked_objects.items():
                    x, y, w, h = obj_data['bbox']
                    label = obj_data.get('label', 'unknown')

                    # Цвета для разных типов объектов
                    colors = {
                        'person': (0, 255, 0),
                        'car': (255, 0, 0),
                        'truck': (0, 0, 255),
                        'default': (255, 255, 255)
                    }
                    text_color = colors.get(label, colors['default'])

                    # Позиция текста
                    text_position = (x + 5, y + h - 5)
                    text_position = (max(5, text_position[0]), max(15, text_position[1]))

                    # Рисуем ID и тип объекта
                    cv2.putText(
                        img=annotated_frame,
                        text=f"{label} ID: {obj_id}",
                        org=text_position,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=text_color,
                        thickness=1,
                        lineType=cv2.LINE_AA
                    )
        else:
            annotated_frame = small_frame.copy()
            tracked_objects = {}  # Пустой словарь, если нет детекций

        # Логирование новых объектов
        if tracked_objects:
            for obj_id, obj_data in tracked_objects.items():
                # Проверяем, был ли объект уже залогирован
                if not obj_data.get('logged', False):
                    # Логируем только один раз при первом обнаружении
                    log_success = log_detection(
                        obj_data.get('label', 'unknown'),  # Используем 'unknown' как резерв
                        obj_data['first_seen'].strftime("%Y-%m-%d %H:%M:%S"),
                        frame_time,
                        obj_id,
                        camera_name
                    )
                    if log_success:
                        obj_data['logged'] = True
        else:
            annotated_frame = small_frame.copy()

        return annotated_frame, tracked_objects
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
    cap = None
    last_check_time = time.time()

    while True:
        try:
            current_time = time.time()

            # Проверяем соединение каждые 10 секунд
            if current_time - last_check_time > 10:
                if not verify_db_entries():
                    print("Проблема с БД")
                last_check_time = current_time

            if cap is None or not cap.isOpened():
                cap = get_video_capture(rtsp_url)
                if cap is None:
                    continue

            ret, frame = cap.read()
            if not ret:
                print(f"Ошибка чтения кадра из {rtsp_url}, переподключение...")
                cap.release()
                cap = None
                continue

            # Обработка кадра
            frame_time = time.time()
            processed_frame, results = process_frame(frame, model, frame_time, camera_name, rotate_180)

            # Отладочный вывод
            if results and hasattr(results[0], 'boxes') and results[0].boxes.id is not None:
                print(f"Камера {camera_name}: Обнаружено {len(results[0].boxes)} объектов")
            else:
                print(f"Камера {camera_name}: Нет объектов для отладки")
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


        except Exception as e:
            print(f"Критическая ошибка в video_processing_thread ({camera_name}): {e}")
            if cap is not None:
                cap.release()
                cap = None


def main():
    st.set_page_config(layout="wide")
    st.title("Мониторинг с детекцией объектов (2 камеры)")

    # Проверяем и создаем БД при запуске
    if not os.path.exists(DB_PATH):
        print("База данных не найдена, создаем новую...")
        init_db()
    else:
        print("База данных уже существует")
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
            if not st.session_state.frame_queue_1.empty():
                item_type, frame = st.session_state.frame_queue_1.get_nowait()
                if item_type == 'frame':
                    # Используем width=0 как замена use_container_width в старых версиях
                    placeholder_1.image(frame, channels="BGR", width=0)
                    status_text_1.success(f"Камера 1: Активна ({datetime.now().strftime('%H:%M:%S')})")
                    last_update_time = current_time

            if not st.session_state.frame_queue_2.empty():
                item_type, frame = st.session_state.frame_queue_2.get_nowait()
                if item_type == 'frame':
                    placeholder_2.image(frame, channels="BGR", width=0)
                    status_text_2.success(f"Камера 2: Активна ({datetime.now().strftime('%H:%M:%S')})")
                    last_update_time = current_time

            # Проверка зависания интерфейса
            if current_time - last_update_time > 5:
                status_text_1.warning("Камера 1: Ожидание кадров...")
                status_text_2.warning("Камера 2: Ожидание кадров...")

        except Exception as e:
            print(f"Ошибка в основном цикле: {e}")


if __name__ == "__main__":
    main()