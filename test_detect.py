from ultralytics import YOLO
import cv2
# Загрузка обученной модели
model = YOLO("C:\\Users\\Admin\\Desktop\\runs result carbrand\\train\\weights\\best.pt")  # Путь к вашей модели
"""
# Инференс на одном изображении
results = model("C:\\Users\\Admin\\Desktop\\MyWork\\images.jpg")

# Вывод результатов
for result in results:
    results[0].show()  # Показать предсказания на изображении
    results[0].plot()  # Получить изображение с bounding boxes (numpy array"""   

# Открытие видеофайла
video_path = "C:\\Users\\Admin\\Desktop\\MyWork\\test4.mp4"
cap = cv2.VideoCapture(video_path)

# Проверка успешного открытия
if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Детектирование
    results = model(frame)

    # Отрисовка результатов
    annotated_frame = results[0].plot()  # Добавляет bounding boxes и метки

    # Отображение кадра
    cv2.imshow("YOLO Inference", annotated_frame)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()