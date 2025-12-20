"""
6404voroninpm_lab5 - Пакет для асинхронной обработки изображений животных.
Группа: 6404, Фамилия: Voronin
"""

__version__ = "1.0.0"
__author__ = "Voronin"
__group__ = "6404"
__description__ = "Асинхронная обработка изображений животных с параллельными процессами и логированием"

# Экспорт публичных классов и функций
from .async_cat_image_processor_v2 import AsyncCatImageProcessorV2
from .async_logging_config import setup_async_logging, get_async_logger
from .cat_image import CatImage, create_cat_image
from .main_lab5 import main_lab5, main_lab5_async

__all__ = [
    'AsyncCatImageProcessorV2',
    'setup_async_logging',
    'get_async_logger',
    'CatImage',
    'create_cat_image',
    'main_lab5',
    'main_lab5_async'
]