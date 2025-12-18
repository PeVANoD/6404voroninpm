"""
lab4_project - Пакет для асинхронной обработки изображений животных.

Этот пакет содержит функционал для:
- Асинхронной загрузки изображений через API
- Обработки изображений с обнаружением контуров
- Параллельной обработки с использованием многопроцессорности
- Логирования и тестирования

Основные компоненты:
- AsyncCatImageProcessor: главный класс для обработки
- CatImage, ColorCatImage, GrayscaleCatImage: классы для работы с изображениями
- Настройка логирования через модуль logging_config
- Тесты в модуле tests
"""

__version__ = "1.0.0"
__author__ = "Ваше Имя"
__description__ = "Асинхронная обработка изображений животных с параллельными процессами"

# Импортируем основные классы для удобного доступа
from .async_cat_image_processor import AsyncCatImageProcessor
from .cat_image import CatImage, ColorCatImage, GrayscaleCatImage, create_cat_image
from .logging_config import setup_logging, get_logger
from .main_lab4 import main_lab4

# Настройка логирования при импорте пакета
logger = setup_logging()

__all__ = [
    'AsyncCatImageProcessor',
    'CatImage',
    'ColorCatImage',
    'GrayscaleCatImage',
    'create_cat_image',
    'setup_logging',
    'get_logger',
    'main_lab4',
    'logger'
]

print(f"Пакет {__name__} версии {__version__} успешно загружен")