import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO
from datetime import datetime
import ollama
import tempfile
from PIL import Image

def analyze_frame_with_llava(frame):
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmpfile:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img.save(tmpfile.name, format='JPEG')
            tmpfile_path = tmpfile.name  # Сохраняем путь
        # Теперь файл закрыт, и мы можем его использовать
        response = ollama.chat(
            model="llava",
            messages=[{
                "role": "user",
                "content": "Describe what you see in this frame?",
                "images": [tmpfile_path]
            }]
        )
        os.unlink(tmpfile_path)  # Удаляем файл после использования
        return response["message"]["content"]
    except Exception as e:
        print(f"Ошибка анализа через LLaVA: {e}")
        return "Ошибка анализа"

class UniqueObjectTracker:
    def __init__(self, max_age=10):
        self.active_objects = {}  # {obj_id: {'bbox': [x,y,w,h], 'first_seen': timestamp}}
        self.max_age = max_age
        self.next_id = 0

    def register(self, bbox):
        self.active_objects[self.next_id] = {
            'bbox': bbox,
            'first_seen': datetime.now(),
            'last_seen': datetime.now(),
            'analyzed': False
        }
        obj_id = self.next_id
        self.next_id += 1
        return obj_id

    def update(self, detections):
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
                        'last_seen': datetime.now(),
                        'label': label,
                        'analyzed': obj_data['analyzed']
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
                    'analyzed': False
                }

        # Удаление старых объектов
        for obj_id in list(self.active_objects.keys()):
            if obj_id not in updated:
                if (datetime.now() - self.active_objects[obj_id]['last_seen']).total_seconds() < self.max_age:
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

# === Параметры ===
VIDEO_PATH = "C:\\Users\\Admin\\Desktop\\MyWork\\test5.mp4"
MODEL_NAME = "llava"
CONFIDENCE_THRESHOLD = 0.5
MAX_AGE = 10  # Количество кадров для хранения "исчезнувшего" объекта

# === Инициализация ===
cap = cv2.VideoCapture(VIDEO_PATH)
model = YOLO('C:\\Users\\Admin\\Desktop\\MyWork\\yolo11n.pt')  # Убедитесь, что модель доступна
tracker = UniqueObjectTracker(max_age=MAX_AGE)

# === Фоновый субтрактор ===
back_sub = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # === Детекция движения ===
    fg_mask = back_sub.apply(frame)
    _, mask_thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

    if large_contours:
        results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
        if results and hasattr(results[0], 'boxes') and results[0].boxes.id is not None:
            detections = []
            for obj in results[0].boxes:
                x1, y1, x2, y2 = obj.xyxy[0].tolist()
                bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                label = model.names[int(obj.cls)]
                detections.append({'bbox': bbox, 'label': label})

            tracked_objects = tracker.update(detections)

            # === Обработка найденных объектов ===
            for obj_id, obj_data in tracked_objects.items():
                if not obj_data.get('analyzed', False):
                    x, y, w, h = obj_data['bbox']
                    obj_frame = frame[y:y+h, x:x+w]

                    # === Запрос к LLaVA ===
                    description = analyze_frame_with_llava(obj_frame)
                    print(f"Объект ID: {obj_id}, Тип: {obj_data['label']}, Описание: {description}")

                    # === Помечаем как обработанный ===
                    obj_data['analyzed'] = True
                    tracker.active_objects[obj_id] = obj_data

            # === Рисуем боксы ===
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame.copy()
    else:
        annotated_frame = frame.copy()

    # === Отображение ===
    cv2.imshow("Видеоанализ", annotated_frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()