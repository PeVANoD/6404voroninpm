"""
config.py

Конфигурационные параметры для лабораторных работ.
"""

import os

# API настройки
API_KEY = API_KEY = os.getenv('CAT_API_KEY', 'live_QSP9D2z1VT6qPX0N8xaqUBIN89o6nO4S7KKNunpJFSv7GKI37Y2LOGiiNpyVTE2e')
BASE_URL = "https://api.thecatapi.com/v1"

# Настройки по умолчанию
DEFAULT_LIMIT = 3
DEFAULT_OUTPUT_DIR = "processed_images_async"  # Это значение используется по умолчанию

# Для lab5 можно добавить отдельную константу
DEFAULT_OUTPUT_DIR_V2 = "processed_images_async_v2"