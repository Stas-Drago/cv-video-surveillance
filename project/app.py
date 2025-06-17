import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Flask, render_template, Response, request, jsonify
import cv2
import threading
from ultralytics import YOLO
from datetime import datetime
import ollama
import tempfile
from PIL import Image
import time
from datetime import datetime, timedelta
import sqlite3

app = Flask(__name__)

# === Настройки камер ===
CAMERAS = {
    "Камера 1": "http://192.168.5.35:555/d3s1uCw2?container=mjpeg&stream=main",
    "Камера 2": "http://192.168.5.35:555/JZGVXGWS?container=mjpeg&stream=main"
}

# === Глобальные переменные ===
current_camera_url = list(CAMERAS.values())[0]
latest_frame = None
lock = threading.Lock()
user_question = "Describe what you see in this frame?"
question_lock = threading.Lock()
force_reconnect = False
reconnect_lock = threading.Lock()

# === Загрузка моделей ===
model = YOLO('yolo11n.pt')  # Убедитесь, что модель доступна
back_sub = cv2.createBackgroundSubtractorMOG2()

# === Класс трекера (без изменений) ===
class UniqueObjectTracker:
    def __init__(self, max_age=10):
        self.active_objects = {}
        self.max_age = max_age
        self.next_id = 0

    def register(self, bbox):
        self.active_objects[self.next_id] = {
            'bbox': bbox,
            'first_seen': datetime.now(),
            'last_seen': datetime.now(),
            'analyzed': False,
            'type': None
        }
        obj_id = self.next_id
        self.next_id += 1
        return obj_id

    def update(self, detections):
        updated = {}
        for det_bbox, label in detections:
            matched = False
            for obj_id, obj_data in self.active_objects.items():
                if self.iou(det_bbox, obj_data['bbox']) > 0.3:
                    updated[obj_id] = {
                        'bbox': det_bbox,
                        'first_seen': obj_data['first_seen'],
                        'last_seen': datetime.now(),
                        'label': label,
                        'analyzed': obj_data['analyzed'],
                        'type': obj_data['type']
                    }
                    matched = True
                    break
            if not matched:
                new_id = self.register(det_bbox)
                updated[new_id] = {
                    'bbox': det_bbox,
                    'first_seen': datetime.now(),
                    'last_seen': datetime.now(),
                    'label': label,
                    'analyzed': False,
                    'type': label
                }
        now = datetime.now()
        for obj_id in list(self.active_objects.keys()):
            if obj_id not in updated and (now - self.active_objects[obj_id]['last_seen']).total_seconds() < self.max_age:
                updated[obj_id] = self.active_objects[obj_id]
        self.active_objects = updated
        return self.active_objects

    def iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1 + w1, x2 + w2)
        yB = min(y1 + h1, y2 + h2)
        interArea = max(0, xB - xA) * max(0, yB - yA)
        box1Area = w1 * h1
        box2Area = w2 * h2
        return interArea / (box1Area + box2Area - interArea)

# === Глобальные переменные для анализа ===
active_analysis_objects = {}  # Хранение объектов, которые нужно проанализировать
tracker = UniqueObjectTracker(max_age=10)

# === Создание БД и таблицы ===
def init_db():
    conn = sqlite3.connect('surveillance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  camera TEXT,
                  obj_type TEXT,
                  obj_id INTEGER,
                  answer TEXT)''')
    conn.commit()
    conn.close()

init_db()

# === Логирование в БД ===
def log_analysis(obj_id, obj_type, description):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    camera_name = next(key for key, value in CAMERAS.items() if value == current_camera_url)
    conn = sqlite3.connect('surveillance.db')
    c = conn.cursor()
    c.execute("INSERT INTO logs (timestamp, camera, obj_type, obj_id, answer) VALUES (?, ?, ?, ?, ?)",
              (timestamp, camera_name, obj_type, obj_id, description))
    conn.commit()
    conn.close()

# === Анализ кадра через LLaVA ===
def analyze_frame_with_llava(frame, question="Describe what you see in this frame?"):
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmpfile:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img.save(tmpfile.name, format='JPEG')
            tmpfile_path = tmpfile.name
        response = ollama.chat(
            model="llava",
            messages=[{
                "role": "user",
                "content": question,
                "images": [tmpfile_path]
            }]
        )
        os.unlink(tmpfile_path)
        return response.get("message", {}).get("content", "Нет содержания в ответе")
    except Exception as e:
        print(f"Ошибка анализа через LLaVA: {e}")
        return f"Ошибка анализа: {e}"

# === Поток для чтения кадров и анализа ===
def video_processing_thread():
    global current_camera_url, latest_frame, force_reconnect
    cap = None
    while True:
        with reconnect_lock:
            if force_reconnect or (cap is None or not cap.isOpened()):
                if cap is not None:
                    cap.release()
                print(f"Подключились к: {current_camera_url}")
                cap = cv2.VideoCapture(current_camera_url)
                force_reconnect = False  # ✅ Сбрасываем флаг
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения кадра. Переподключение...")
            cap.release()
            cap = None
            time.sleep(5)
            continue
        # === ВСЕГДА запускаем YOLO ===
        results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
        detections = []
        if results and hasattr(results[0], 'boxes'):
            for obj in results[0].boxes:
                x1, y1, x2, y2 = obj.xyxy[0].tolist()
                bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                label = model.names[int(obj.cls)]
                detections.append((bbox, label))
        tracked_objects = tracker.update(detections)
        # === Анализ новых объектов ===
        for obj_id, obj_data in tracked_objects.items():
            if not obj_data.get('analyzed', False):
                x, y, w, h = obj_data['bbox']
                obj_frame = frame[y:y+h, x:x+w]
                with question_lock:
                    description = analyze_frame_with_llava(obj_frame, user_question)
                log_analysis(obj_id, obj_data['label'], description)
                obj_data['analyzed'] = True
                tracker.active_objects[obj_id] = obj_data
        if results and hasattr(results[0], 'boxes'):
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame.copy()
        with lock:
            latest_frame = annotated_frame
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index.html', cameras=CAMERAS.keys())

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_camera', methods=['POST'])
def set_camera():
    global current_camera_url, force_reconnect
    with reconnect_lock:
        camera_name = request.json.get('camera')
        if camera_name in CAMERAS:
            current_camera_url = CAMERAS[camera_name]
            force_reconnect = True  # ✅ Активируем переподключение
            return jsonify({"status": "success", "camera": camera_name})
        return jsonify({"status": "error", "message": "Camera not found"}), 400

@app.route('/search_logs', methods=['GET'])
def search_logs():
    time_filter = request.args.get('time', '')
    start_time = request.args.get('start_time', '')
    end_time = request.args.get('end_time', '')
    obj_type = request.args.get('type', '')
    keyword = request.args.get('keyword', '').lower()
    camera = request.args.get('camera', '')

    query = "SELECT * FROM logs WHERE 1=1"
    params = []

    if time_filter:
        try:
            minutes = int(time_filter)
            cutoff = datetime.now() - timedelta(minutes=minutes)
            query += " AND timestamp >= ?"
            params.append(cutoff.strftime("%Y-%m-%d %H:%M:%S"))
        except ValueError:
            pass

    if start_time:
        query += " AND timestamp >= ?"
        params.append(start_time)
    if end_time:
        query += " AND timestamp <= ?"
        params.append(end_time)

    if obj_type:
        query += " AND obj_type = ?"
        params.append(obj_type)

    if keyword:
        query += " AND answer LIKE ?"
        params.append(f"%{keyword}%")

    if camera:
        query += " AND camera = ?"
        params.append(camera)

    conn = sqlite3.connect('surveillance.db')
    c = conn.cursor()
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()

    logs = [{"timestamp": r[1], "camera": r[2], "type": r[3], "obj_id": r[4], "answer": r[5]} for r in rows]
    logs.sort(key=lambda x: x['timestamp'], reverse=True)
    return jsonify(logs)

# === MJPEG поток ===
def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

if __name__ == '__main__':
    threading.Thread(target=video_processing_thread, daemon=True).start()
    app.run(host='0.0.0.0', port=8090, debug=True)