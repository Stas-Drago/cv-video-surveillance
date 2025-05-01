import torch
print(f"CUDA доступен: {torch.cuda.is_available()}")
print(f"Количество GPU: {torch.cuda.device_count()}")
print(f"Текущее устройство: {torch.cuda.current_device()}")
print(f"Имя устройства: {torch.cuda.get_device_name(torch.cuda.current_device())}")