from ultralytics import YOLO
import cv2
import numpy as np

# === Шаг 1: Загрузка модели YOLOv8 ===
model = YOLO("yolov8n.pt")  # Используйте yolov8n.pt или другую предобученную модель

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
def input_polygon():
    """
    Запрашивает у пользователя координаты вершин полигона через терминал.
    :return: Список координат вершин полигона
    """
    print("Введите координаты вершин полигона (x, y). Для завершения ввода введите 'done'.")
    polygon = []
    while True:
        user_input = input(f"Введите координаты вершины {len(polygon) + 1} (например, 'x,y'): ")
        if user_input.lower() == "done":
            break
        try:
            x, y = map(int, user_input.split(","))
            polygon.append([x, y])
        except ValueError:
            print("Ошибка: Неверный формат координат. Введите координаты в формате 'x,y'.")
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
    polygon = input_polygon()

    # Создание маски двора
    mask = create_courtyard_mask((height, width), polygon)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: Связь с камерой потеряна.")
            break

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