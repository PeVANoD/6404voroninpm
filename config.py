"""
Конфигурационный файл для работы с API.
"""
import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла с указанием кодировки
try:
    load_dotenv(encoding='utf-8')
except:
    # Если возникает ошибка с кодировкой, пробуем без нее
    try:
        load_dotenv()
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")

# Параметры API
API_KEY = os.getenv('CAT_API_KEY', 'demo_api_key')  # Запасной ключ для демо
BASE_URL = 'https://api.thecatapi.com/v1'

# Параметры по умолчанию
DEFAULT_LIMIT = 3
DEFAULT_OUTPUT_DIR = 'processed_images_async'