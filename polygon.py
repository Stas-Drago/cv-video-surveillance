from ultralytics import YOLO
import cv2
import numpy as np
import torch

# === Шаг 1: Загрузка модели YOLOv8 ===
# В загрузке модели укажите устройство
model = YOLO("yolov8m.pt").to('cuda' if torch.cuda.is_available() else 'cpu')  # Используйте yolov8n.pt или другую предобученную модель

# === Шаг 2: Определение маски двора ===
def create_courtyard_mask(image_shape, polygon):
    """
    Создает маску двора на основе заданного полигона.
    :param image_shape: Размеры изображения (высота, ширина)
    :param polygon: Координаты вершин полигона (список точек)
    :return: Маска двора (numpy array)
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], color=255)
    return mask

# === Шаг 3: Обработка видео или изображения ===
def process_frame(frame, mask, polygon):
    """
    Обрабатывает кадр: применяет маску и детектирует объекты.
    :param frame: Кадр (numpy array)
    :param mask: Маска двора (numpy array)
    :param polygon: Координаты вершин полигона (список точек)
    :return: Обработанный кадр с отмеченными объектами
    """
    # Детектирование объектов
    results = model(frame)

    for result in results:
        boxes = result.boxes.cpu().numpy()  # Получаем bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты bounding box
            cls = int(box.cls[0])  # Класс объекта (0 - человек, 2 - машина)

            # Проверяем, находится ли центр объекта внутри маски
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            if mask[center_y, center_x] == 255:  # Если центр внутри маски
                label = "Person" if cls == 0 else "Car"
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # Зеленый для человека, красный для машины
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Отрисовка границ двора
    cv2.polylines(frame, [np.array(polygon, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

    return frame

# === Шаг 4: Ввод координат границ через терминал ===
def select_polygon(image):
    """
    Позволяет пользователю нарисовать полигон на изображении
    :param image: Исходное изображение
    :return: Список точек полигона
    """
    polygon = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            polygon.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Draw Polygon", image)
            
    cv2.namedWindow("Draw Polygon")
    cv2.setMouseCallback("Draw Polygon", mouse_callback)
    
    while True:
        cv2.imshow("Draw Polygon", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
    cv2.destroyAllWindows()
    return polygon
# === Шаг 5: Основной цикл программы ===
def main():
    # RTSP-адрес камеры
    rtsp_url = "rtsp://user:0704RRrr@192.168.4.65:554/Streaming/Channels/101"

    # Подключение к камере
    cap = cv2.VideoCapture(rtsp_url)

    # Проверка подключения
    if not cap.isOpened():
        print("Ошибка: Не удалось подключиться к камере.")
        return

    # Чтение первого кадра для определения размеров
    ret, first_frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось получить первый кадр от камеры.")
        return

    # Вывод размера изображения
    height, width = first_frame.shape[:2]
    print(f"Размер изображения: {width}*{height} пикселей")

    # Ввод координат границ через терминал
    polygon = select_polygon(first_frame.copy())

    # Создание маски двора
    mask = create_courtyard_mask((height, width), polygon)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Потеряно соединение, пытаемся переподключиться...")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            continue
        # ... обработка ...

        # Обработка кадра
        processed_frame = process_frame(frame, mask, polygon)

        # Отображение результата
        cv2.imshow("Detected Objects", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()