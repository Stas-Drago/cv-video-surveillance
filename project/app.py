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

app = Flask(__name__)

# === Настройки камер ===
CAMERAS = {
    "Камера 1": "http://192.168.5.35:555/d3s1uCw2?container=mjpeg&stream=main",
    "Камера 2": "http://192.168.5.35:555/JZGVXGWS?container=mjpeg&stream=main",
    "Камера 3": "http://192.168.5.35:555/JZGVXGWS?container=mjpeg&stream=main"
}

# === Глобальные переменные ===
current_camera_url = list(CAMERAS.values())[0]
latest_frame = None
lock = threading.Lock()
user_question = "Describe what you see in this frame?"
question_lock = threading.Lock()

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
analysis_log_file = "analysis_log.txt"
tracker = UniqueObjectTracker(max_age=10)
# === Анализ кадра через LLaVA ===
def analyze_frame_with_llava(frame, question="Describe what you see in this frame?"):
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmpfile:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img.save(tmpfile.name, format='JPEG')
            tmpfile_path = tmpfile.name

        print(f"Отправляем вопрос модели: {question}")
        response = ollama.chat(
            model="llava",
            messages=[{
                "role": "user",
                "content": question,
                "images": [tmpfile_path]
            }]
        )
        print("Ответ модели:", response)

        os.unlink(tmpfile_path)
        return response.get("message", {}).get("content", "Нет содержания в ответе")
    except Exception as e:
        print(f"Ошибка анализа через LLaVA: {e}")
        return f"Ошибка анализа: {e}"

# === Поток для чтения кадров и анализа ===
def video_processing_thread():
    global current_camera_url, latest_frame, active_analysis_objects, user_question
    cap = None

    while True:
        if cap is None or not cap.isOpened():
            print(f"Попытка подключения к: {current_camera_url}")
            cap = cv2.VideoCapture(current_camera_url)
            print(f"Формат потока: {cap.get(cv2.CAP_PROP_FORMAT)}")
            print(f"Подключились к: {current_camera_url}")

        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения кадра, переподключение...")
            cap.release()
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

                # === Запрос к LLaVA ===
                with question_lock:
                    description = analyze_frame_with_llava(obj_frame, user_question)
                    print(f"Объект ID: {obj_id}, Тип: {obj_data['label']}, Описание: {description}")

                # === Логирование ===
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_entry = f"[{timestamp}] Объект ID: {obj_id}, Тип: {obj_data['label']}, Ответ: {description}\n"
                with open(analysis_log_file, "a", encoding="utf-8") as f:
                    f.write(log_entry)

                # === Помечаем как обработанный ===
                obj_data['analyzed'] = True
                tracker.active_objects[obj_id] = obj_data

        # === Рисуем боксы ===
        if results and hasattr(results[0], 'boxes'):
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame.copy()

        with lock:
            latest_frame = annotated_frame

        time.sleep(0.03)  # Ограничение частоты обработки

@app.route('/')
def index():
    return render_template('index.html', cameras=CAMERAS.keys())

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_camera', methods=['POST'])
def set_camera():
    global current_camera_url
    camera_name = request.json.get('camera')
    if camera_name in CAMERAS:
        current_camera_url = CAMERAS[camera_name]
        return jsonify({"status": "success", "camera": camera_name})
    return jsonify({"status": "error", "message": "Camera not found"}), 400

@app.route('/ask_model', methods=['POST'])
def ask_model():
    global latest_frame
    question = request.json.get('question', "Describe what you see in this frame?")
    
    if latest_frame is None:
        print("Ошибка: latest_frame == None")
        return jsonify({"answer": "Кадр не доступен для анализа"})
    
    try:
        answer = analyze_frame_with_llava(latest_frame, question)
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Ошибка при анализе кадра: {e}")
        return jsonify({"answer": "Произошла ошибка при анализе кадра"})
    
from datetime import datetime, timedelta

@app.route('/get_logs')
def get_logs():
    try:
        filter_param = request.args.get('filter', 'all')
        current_time = datetime.now()

        with open("analysis_log.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            logs = []
            for line in lines:
                if "Объект ID:" in line:
                    parts = line.strip().split("] ")
                    timestamp_str = parts[0].replace("[", "").replace("]", "")
                    rest = parts[1]
                    try:
                        obj_id = rest.split("ID:")[1].split(",")[0].strip()
                        obj_type = rest.split("Тип:")[1].split(",")[0].strip()
                        answer = rest.split("Ответ:")[1].strip()
                    except IndexError:
                        continue

                    # Парсим время события
                    event_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    keep = True

                    if filter_param == "10m":
                        keep = (current_time - event_time) < timedelta(minutes=10)
                    elif filter_param == "1h":
                        keep = (current_time - event_time) < timedelta(hours=1)
                    elif filter_param == "24h":
                        keep = (current_time - event_time) < timedelta(days=1)

                    if keep:
                        logs.append({
                            "timestamp": timestamp_str,
                            "type": obj_type,
                            "id": obj_id,
                            "answer": answer
                        })

            # Сортируем по времени (новые сверху)
            logs.sort(key=lambda x: x['timestamp'], reverse=True)
            return jsonify(logs)
    except Exception as e:
        print(f"Ошибка чтения логов: {e}")
        return jsonify([])
    
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