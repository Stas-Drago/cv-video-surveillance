import ollama
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import tempfile
import os

# === Параметры ===
MODEL_NAME = "llava"
RTSP_URL = "rtsp://user:0704RRrr@192.168.4.65:554/Streaming/Channels/101"

class VideoApp:
    def __init__(self, window):
        self.window = window
        self.window.title("LLaVA + RTSP")
        self.video_running = False
        self.cap = None
        self.current_frame = None

        # === Создание вкладок ===
        self.tab_control = ttk.Notebook(window)
        self.tab_video = ttk.Frame(self.tab_control)
        self.tab_image = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_video, text='Видео')
        self.tab_control.add(self.tab_image, text='Изображение')
        self.tab_control.pack(expand=1, fill="both")

        # === Вкладка "Видео" ===
        self.video_label = tk.Label(self.tab_video)
        self.video_label.pack(pady=10)

        self.btn_start = tk.Button(self.tab_video, text="▶️ Запустить RTSP", command=self.start_video)
        self.btn_start.pack()

        self.question_entry = tk.Entry(self.tab_video, width=50)
        self.question_entry.pack(pady=10)
        self.question_entry.insert(0, "Describe what you see in the image?")

        self.btn_analyze = tk.Button(self.tab_video, text="🧠 Анализировать кадр", command=self.analyze_current_frame)
        self.btn_analyze.pack()

        self.answer_label = tk.Label(self.tab_video, text="", wraplength=500, justify="left")
        self.answer_label.pack(pady=10)

        # === Вкладка "Изображение" ===
        self.image_path = None
        self.image_label = tk.Label(self.tab_image)
        self.image_label.pack(pady=10)

        tk.Button(self.tab_image, text="📁 Выбрать изображение", command=self.select_image).pack()

        self.image_question = tk.Entry(self.tab_image, width=50)
        self.image_question.pack(pady=10)
        self.image_question.insert(0, "Опиши это изображение")

        tk.Button(self.tab_image, text="🧠 Анализировать", command=self.analyze_image).pack()

        self.image_answer = tk.Label(self.tab_image, text="", wraplength=500, justify="left")
        self.image_answer.pack(pady=10)

    # === Видео-функции ===
    def start_video(self):
        if self.video_running:
            self.stop_video()
        
        self.cap = cv2.VideoCapture(RTSP_URL)
        if not self.cap.isOpened():
            messagebox.showerror("Ошибка", "Не удалось подключиться к RTSP-потоку")
            return

        self.video_running = True
        self.update_frame()

    def stop_video(self):
        self.video_running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image=None)

    def update_frame(self):
        if self.video_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                # Конвертация кадра для Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((1200, 800))
                photo = ImageTk.PhotoImage(img)
                self.video_label.config(image=photo)
                self.video_label.image = photo  # Сохраняем ссылку
        self.window.after(30, self.update_frame)  # Обновление каждые 30 мс

    def analyze_current_frame(self):
        if self.current_frame is None:
            messagebox.showerror("Ошибка", "Нет кадра для анализа")
            return
            
        try:
            # Сохранение кадра во временный файл
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmpfile:
                img = Image.fromarray(cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB))
                img.save(tmpfile.name, format='JPEG')
                
                # Запрос к Ollama
                response = ollama.chat(
                    model=MODEL_NAME,
                    messages=[{
                        "role": "user",
                        "content": self.question_entry.get(),
                        "images": [tmpfile.name]
                    }]
                )
                
                self.answer_label.config(text=response["message"]["content"])
                os.unlink(tmpfile.name)  # Удаление временного файла
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось обработать кадр: {e}")

    # === Функции работы с изображением ===
    def select_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if self.image_path:
            img = Image.open(self.image_path)
            img = img.resize((300, 300))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo

    def analyze_image(self):
        if not self.image_path:
            messagebox.showerror("Ошибка", "Загрузите изображение!")
            return
            
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": self.image_question.get(),
                    "images": [self.image_path]
                }]
            )
            self.image_answer.config(text=response["message"]["content"])
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось обработать изображение: {e}")


# === Запуск приложения ===
root = tk.Tk()
app = VideoApp(root)
root.mainloop()