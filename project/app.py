import os
from flask import Flask, render_template, Response, request, jsonify
import cv2
import threading
from ultralytics import YOLO
from datetime import datetime, timedelta
import ollama
import tempfile
from PIL import Image
import time
import sqlite3
import uuid
import logging
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# --- Настройки ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLL_NAME = "surveillance_collection"
MARKDOWN_FILE = "logs.md"
OLLAMA_MODEL = "gemma3:12b-it-qat"
MISTRAL_MODEL = "mistral:7b"
BI_ENCODER_MODEL = "sentence-transformers/LaBSE"

# --- Логирование ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Инициализация клиента Qdrant ---
try:
    qdrant_client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    logger.info("Подключились к Qdrant")
except Exception as e:
    logger.error(f"Не удалось подключиться к Qdrant: {e}")
    exit(1)

# --- Инициализация bi-encoder ---
try:
    bi_encoder = SentenceTransformer(BI_ENCODER_MODEL, device='cpu')
    bi_encoder_dim = bi_encoder.get_sentence_embedding_dimension()
    logger.info(f"Инициализирован bi-encoder LaBSE, размерность: {bi_encoder_dim}")
except Exception as e:
    logger.error(f"Ошибка при инициализации bi-encoder: {e}")
    exit(1)

# --- Инициализация Flask ---
app = Flask(__name__)

# === Настройки камер ===
CAMERAS = {
    "Камера 1": "http://192.168.5.35:555/d3s1uCw2?container=mjpeg&stream=main",
    "Камера 2": "http://192.168.5.35:555/JZGVXGWS?container=mjpeg&stream=main",
    "Камера 3": "http://192.168.5.35:555/PEWIsThO?container=mjpeg&stream=main",
    "Камера 4": "http://192.168.5.35:555/YBFbIRT0?container=mjpeg&stream=main"
}

# === Глобальные переменные ===
camera_threads = {}
latest_frames = {}
lock_objects = {camera: threading.Lock() for camera in CAMERAS}
force_reconnect_flags = {camera: False for camera in CAMERAS}
reconnect_lock = threading.Lock()
user_question = "Опиши, что ты видишь на этом кадре? Напиши кратко, описывай только ключевые моменты."
question_lock = threading.Lock()

# === Глобальный трекер (для отслеживания объектов между камерами)
global_tracker = {}

# === Загрузка моделей ===
model = YOLO('yolo11n.pt')
back_sub = cv2.createBackgroundSubtractorMOG2()

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
                  description TEXT)''')
    conn.commit()
    conn.close()
init_db()

# === Логирование в БД и Markdown ===
def log_analysis(camera_name, obj_id, obj_type, description):
    global global_tracker

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect('surveillance.db')
    c = conn.cursor()
    # Проверяем, не дублируется ли obj_id на той же камере
    c.execute("SELECT COUNT(*) FROM logs WHERE obj_id=? AND camera=?", (obj_id, camera_name))
    exists = c.fetchone()[0] > 0
    if exists:
        conn.close()
        return

    # Добавляем новую запись
    c.execute("INSERT INTO logs (timestamp, camera, obj_type, obj_id, answer) VALUES (?, ?, ?, ?, ?)",
              (timestamp, camera_name, obj_type, obj_id, description))
    conn.commit()
    conn.close()

    # Обновляем global_tracker
    if obj_id not in global_tracker:
        global_tracker[obj_id] = {
            "first_seen": timestamp,
            "cameras": []
        }
    global_tracker[obj_id]["cameras"].append({
        "camera": camera_name,
        "timestamp": timestamp,
        "description": description
    })

    # Запись в Markdown
    with open(MARKDOWN_FILE, 'a+', encoding='utf-8') as f:
        f.write(f"\n---\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Camera: {camera_name}\n")
        f.write(f"Type: {obj_type}\n")
        f.write(f"ID: {obj_id}\n")
        f.write(f"Description: {description}\n")

    # Отправка каждые 5 минут
    check_and_send_to_qdrant()

# === Таймер отправки в Qdrant ===
last_send_time = datetime.now()

def check_and_send_to_qdrant():
    global last_send_time
    now = datetime.now()
    if (now - last_send_time).total_seconds() >= 300:  # 5 минут
        process_and_send_to_qdrant()
        last_send_time = now

# === Отправка чанков в Qdrant ===
def process_and_send_to_qdrant():
    try:
        with open(MARKDOWN_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        entries = content.strip().split('\n---\n')
        chunks = []
        metadata_list = []

        for entry in entries:
            if not entry.strip():
                continue
            lines = entry.strip().split('\n')
            meta = {}
            desc = ""
            for line in lines:
                if line.startswith("Timestamp:"):
                    meta['timestamp'] = line.split(":", 1)[1].strip()
                elif line.startswith("Camera:"):
                    meta['camera'] = line.split(":", 1)[1].strip()
                elif line.startswith("Type:"):
                    meta['type'] = line.split(":", 1)[1].strip()
                elif line.startswith("ID:"):
                    meta['id'] = int(line.split(":", 1)[1].strip())
                elif line.startswith("Description:"):
                    desc = line.split(":", 1)[1].strip()
            if desc:
                chunks.append(desc)
                metadata_list.append(meta)

        if len(chunks) == 0:
            return

        # Разбиваем на чанки по 7 записей
        chunk_size = 7
        for i in range(0, len(chunks), chunk_size):
            batch_chunks = chunks[i:i+chunk_size]
            batch_metadata = metadata_list[i:i+chunk_size]

            embeddings = bi_encoder.encode(batch_chunks, convert_to_tensor=True).tolist()

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "chunk": chunk,
                        "metadata": metadata
                    }
                )
                for chunk, embedding, metadata in zip(batch_chunks, embeddings, batch_metadata)
            ]

            qdrant_client.upsert(collection_name=COLL_NAME, wait=True, points=points)

        # Очищаем файл
        with open(MARKDOWN_FILE, 'w', encoding='utf-8') as f:
            f.write('')
        logger.info(f"{len(chunks)} записей успешно отправлено в Qdrant")

    except Exception as e:
        logger.error(f"Ошибка при отправке данных в Qdrant: {e}")

# === Анализ кадра через Ollama ===
def analyze_frame_with_ollama(frame, camera_name=None, obj_id=None, obj_type=None):
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmpfile:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img.save(tmpfile.name, format='JPEG')
            tmpfile_path = tmpfile.name

        system_prompt = f"""# Ты — видеоаналитическая система безопасности. 
Твоя задача — обрабатывать входящие кадры с камер видеонаблюдения, анализировать происходящее в кадре и предоставлять структурированное описание ключевых событий.
### У тебя есть доступ к двум камерам:
Камера 1: парковка во дворе компании.
Камера 2: пропускной пункт внутри здания.
Камера 3 и 4: направлены на коридор внутри офиса компании

### Твои правила:
1. **Не выдумывай** то, что не видно на кадре.
2. **Не предполагай мотивы**, намерения или действия, если они не зафиксированы явно.
3. **Не используй домыслы** о прошлом или будущем событиях.
4. Описывай **только визуальные элементы**.
5. Если сложно определить детали — сообщи об этом честно.

### Метаданные кадра:
- Камера: {camera_name if camera_name else 'неизвестна'}
- ID объекта: {obj_id if obj_id is not None else 'не задан'}
- Время анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Что нужно сделать:
1. Уточни, с какой камеры сделан кадр (используй метаданные).
2. Опиши **только один объект**, который ты видишь на изображении.
3. Укажи его ID из метаданных.
4. Перечисли ключевые действия или события, связанные с этим объектом.
5. Сообщите, если объект не определён, или нет активности.
6. Укажи возможные аномалии, если они есть.
7. Ответ должен быть точным, лаконичным и начинаться с указания номера камеры.

### Формат ответа:
Камера X 
Описание: [основное описание кадра с написанием ID] 
События:  
- [событие 1]  
- [событие 2]  
Аномалии: [краткая информация о возможных подозрительных действиях или отклонениях]

### Пример ответа:
Камера 1  
Описание: На парковке находится один автомобиль марки Toyota Camry c ID 23. Водитель вышел и направляется к зданию.  
События:  
- Водитель Toyota Camry выходит из машины  
- Водитель направляется к главному входу  
Аномалии: Никаких подозрительных действий не зафиксировано

Камера 2  
Описание: В пропускном пункте находится один сотрудник охраны ID 22 и двое сотрудников компании ID 45 и ID 46.  
События:  
- Сотрудник предъявляет пропуск  
- Ожидание проверки второго сотрудника  
Аномалии: Нет несанкционированных лиц, всё в рамках нормы
"""

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": "Проанализируй кадр и предоставь информацию согласно формату.", "images": [tmpfile_path]}
            ]
        )
        os.unlink(tmpfile_path)

        return response.get("message", {}).get("content", "Нет содержания в ответе")

    except Exception as e:
        logger.error(f"Ошибка анализа через Ollama: {e}")
        return f"Ошибка анализа: {e}"

# === Поиск в Qdrant и ответ от модели Mistral 7B ===
@app.route('/ask_model', methods=['POST'])
def ask_model():
    data = request.get_json()
    query = data.get("question", "")
    current_time = data.get("time", "")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if not current_time else current_time
    if not query:
        return jsonify({"error": "No question provided"}), 400

    try:
        query_emb = bi_encoder.encode(query).tolist()
        results = qdrant_client.search(
            collection_name=COLL_NAME,
            query_vector=query_emb,
            limit=3
        )

        context = "\n".join([hit.payload['chunk'] for hit in results])
        prompt = f"""
Ты — виртуальный ассистент охранника , установленный на контрольно-наблюдательном посту. Твоя задача — помогать охраннику следить за безопасностью на территории, анализировать видеопоток с камер, фиксировать подозрительные события, отвечать на вопросы по обстановке и предоставлять краткие логические выводы.
Ты подключен к системе видеонаблюдения и можешь описывать происходящее на любой из камер, анализировать изменения в обстановке, выделять важные события и своевременно информировать персонал


Основные функции:
Отслеживание изменений в зоне наблюдения.
Краткое описание событий и действий людей/объектов.
Фиксация необычной активности (например: подозрительное лицо, шум ночью, движение в запретной зоне).
Обработка запросов типа:
1. «Что происходит на камере 3?»
2. «Какие события были за последние 10 минут?»
3. «Есть ли движения у склада?»
4. «Покажи последнюю запись с главного входа»
Предоставление структурированной информации без лишних деталей.
Своевременное предупреждение о возможных рисках.


Стиль общения:
Четкий, официально-нейтральный.
Без эмоций, только факты.
Ответы должны быть структурированными и понятными.
При необходимости использовать маркированные списки или тайм-метки.


Примеры вопросов и ответов:
Вопрос:
Какие события произошли за последние 10 минут?

Ответ:
Сейчас 2025-07-09 12:16:11. За последние 10 минут зафиксированы следующие события:

08:42 — Подъезд автомобиля к зданию. Водитель вышел, поговорил с сотрудником, уехал.
08:49 — Прохожий вошёл в зону камеры 7, сделал несколько фотографий и покинул территорию.
08:55 — На камере 4 замечено движение в прилегающем кустарнике. Возможно животное.
Вопрос:
Что ты видишь на камере 2?

Ответ:
Сейчас 2025-02-19 02:55:05. На камере 2 (вход со стороны парковки) наблюдается:

Два сотрудника входят в здание через центральный вход.
Один автомобиль медленно проезжает мимо, номер распознан: А123ВР.
Никаких подозрительных действий не зафиксировано.

Вопрос:
Были ли попытки несанкционированного доступа?

Ответ:
Сейчас 2023-05-23 14:36:40. Да. В 08:36 была попытка открыть дверь склада без авторизации. Лицо не распознано. Система отправила сигнал тревоги на пульт охраны.

Вопрос:
Что происходит у заднего выхода?

Ответ:
На камере 5 (задний выход) — никого нет. Дверь закрыта. Последнее перемещение было зафиксировано в 08:15 — сотрудник вышел на короткое время и вернулся.

Ограничения:
Не выдумывай информацию, если данные отсутствуют.
Если событие неизвестно, сообщи об этом честно.
Не делай предположений без явных признаков.
Не используй сложную терминологию, говори понятно и лаконично.

Контекст:
==========
{context}
==========

Текущее время:
{now}

Вопрос:
==========
{query}
==========
"""

        response = ollama.chat(model=MISTRAL_MODEL, messages=[
            {"role": "system", "content": "Вы — полезный ассистент."},
            {"role": "user", "content": prompt}
        ])

        return jsonify({"answer": response['message']['content']})
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {e}")
        return jsonify({"error": "Internal server error"}), 500

# === Поток для обработки одной камеры ===
def camera_processing_thread(camera_name, camera_url):
    local_model = YOLO('yolo11n.pt')
    tracker = UniqueObjectTracker(max_age=10)
    cap = None
    while True:
        with reconnect_lock:
            if force_reconnect_flags[camera_name] or (cap is None or not cap.isOpened()):
                if cap is not None:
                    cap.release()
                print(f"[{camera_name}] Подключились к: {camera_url}")
                cap = cv2.VideoCapture(camera_url)
                force_reconnect_flags[camera_name] = False

        ret, frame = cap.read()
        if not ret:
            print(f"[{camera_name}] Ошибка чтения кадра. Переподключение...")
            cap.release()
            cap = None
            time.sleep(5)
            continue

        results = local_model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml", conf=0.2)
        detections = []
        if results and hasattr(results[0], 'boxes'):
            for obj in results[0].boxes:
                x1, y1, x2, y2 = obj.xyxy[0].tolist()
                bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                label = local_model.names[int(obj.cls)]
                detections.append((bbox, label))

        tracked_objects = tracker.update(detections)

        # === Анализ новых объектов ===
        for obj_id, obj_data in tracked_objects.items():
            if not obj_data.get('analyzed', False):
                x, y, w, h = obj_data['bbox']
                obj_frame = frame[y:y+h, x:x+w]
                with question_lock:
                    description = analyze_frame_with_ollama(obj_frame, camera_name=camera_name, obj_id=obj_id, obj_type=obj_data['label'])
                log_analysis(camera_name, obj_id, obj_data['label'], description)
                obj_data['analyzed'] = True
                tracker.active_objects[obj_id] = obj_data

        if results and hasattr(results[0], 'boxes'):
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame.copy()
        with lock_objects[camera_name]:
            latest_frames[camera_name] = annotated_frame
        time.sleep(0.03)

# === Flask маршруты ===
@app.route('/')
def index():
    return render_template('index.html', cameras=CAMERAS.keys())

@app.route('/video_feed')
def video_feed():
    camera_name = request.args.get('camera', '')
    if camera_name not in CAMERAS:
        return "Camera not found", 404

    def generate(camera):
        while True:
            frame = latest_frames.get(camera, None)
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)
    return Response(generate(camera_name), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_camera', methods=['POST'])
def set_camera():
    camera_name = request.json.get('camera')
    if camera_name in CAMERAS:
        force_reconnect_flags[camera_name] = True
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

# === Создание коллекции в Qdrant ===
try:
    collections = qdrant_client.get_collections().collections
    collection_names = [col.name for col in collections]
    if COLL_NAME not in collection_names:
        qdrant_client.create_collection(
            collection_name=COLL_NAME,
            vectors_config=VectorParams(size=bi_encoder_dim, distance=Distance.COSINE)
        )
        logger.info(f"Коллекция '{COLL_NAME}' создана в Qdrant")
    else:
        logger.info(f"Коллекция '{COLL_NAME}' уже существует")
except Exception as e:
    logger.error(f"Ошибка проверки коллекции Qdrant: {e}")

# === Класс трекера ===
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
            for obj_id, obj_data in list(self.active_objects.items()):
                if self.iou(det_bbox, obj_data['bbox']) > 0.3:
                    updated[obj_id] = {
                        'bbox': det_bbox,
                        'first_seen': obj_data['first_seen'],
                        'last_seen': datetime.now(),
                        'label': label,
                        'analyzed': obj_data['analyzed'],
                        'type': label
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
        box2Area = w2 * h1
        union = box1Area + box2Area - interArea
        return interArea / union if union != 0 else 0

# === Запуск камер ===
for camera_name, camera_url in CAMERAS.items():
    thread = threading.Thread(
        target=camera_processing_thread,
        args=(camera_name, camera_url),
        daemon=True
    )
    thread.start()
    camera_threads[camera_name] = thread

# === Запуск Flask ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090, debug=True)