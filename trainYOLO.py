from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("C:\\Users\\Admin\\Desktop\\MyWork\\yolo11n.pt")    # Для дообучения предобученной модели

    # Обучение
    results = model.train(
        data="C:\\Users\\Admin\\Desktop\\cars\\data.yaml",     # Путь к вашему data.yaml
        epochs=50,                   # Количество эпох
        imgsz=640,                    # Размер изображения
        device=0,                     # Использование GPU (0 - первый GPU)
        workers=12,                    # Количество ядер CPU
        project="C:\\Users\\Admin\\Desktop\\runs result carbrand",         # Папка для сохранения результатов
        batch=16,
        exist_ok=True,                 # Перезаписывать существующую папку
        amp=False,
    )