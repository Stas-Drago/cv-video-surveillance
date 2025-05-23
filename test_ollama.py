import ollama
from tkinter import Tk, Button, Label, Entry, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

# === Параметры ===
model_name = "llava"  # Имя модели в Ollama
image_path = None

# === Функция анализа изображения ===
def analyze_image():
    global image_path
    
    if not image_path:
        messagebox.showerror("Ошибка", "Загрузите изображение!")
        return
    
    try:
        # Отправка запроса в Ollama
        response = ollama.chat(
            model=model_name,
            messages=[{
                "role": "user",
                "content": entry_question.get(),
                "images": [image_path]
            }]
        )
        
        # Обновление ответа
        label_answer.config(text=response["message"]["content"])
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось обработать изображение: {e}")

# === Функция выбора изображения ===
def select_image():
    global image_path
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    
    if image_path:
        # Отображение изображения в интерфейсе
        img = Image.open(image_path)
        img = img.resize((300, 300))  # Уменьшение размера для отображения
        photo = ImageTk.PhotoImage(img)
        label_image.config(image=photo)
        label_image.image = photo  # Для предотвращения garbage collection

# === Создание интерфейса ===
root = Tk()
root.title("LLaVA via Ollama")
root.geometry("600x500")

# === Элементы интерфейса ===
label_instruction = Label(root, text="Загрузите изображение и задайте вопрос:")
label_instruction.pack(pady=10)

# Кнопка для выбора изображения
btn_select_image = Button(root, text="Выбрать изображение", command=select_image)
btn_select_image.pack()

# Место для отображения изображения
label_image = Label(root)
label_image.pack(pady=10)

# Ввод вопроса
entry_question = Entry(root, width=50)
entry_question.pack(pady=10)

# Кнопка анализа
btn_analyze = Button(root, text="Анализировать", command=analyze_image)
btn_analyze.pack(pady=5)

# Ответ модели
label_answer = Label(root, text="", wraplength=500)
label_answer.pack(pady=10)

# Запуск интерфейса
root.mainloop()