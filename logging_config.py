"""
logging_config.py

Конфигурация логирования для лабораторной работы №4.
Создает два обработчика: файловый (DEBUG) и консольный (INFO).
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(log_file: str = "app.log", 
                  console_level: int = logging.INFO,
                  file_level: int = logging.DEBUG) -> logging.Logger:
    """
    Настройка и возврат логгера для приложения.
    
    Args:
        log_file: Имя файла для логов
        console_level: Уровень логирования для консоли
        file_level: Уровень логирования для файла
    
    Returns:
        Сконфигурированный логгер
    """
    # Создаем логгер с именем нашего приложения
    logger = logging.getLogger("lab4_async_processor")
    logger.setLevel(logging.DEBUG)  # Ловим все сообщения
    
    # Очищаем существующие обработчики (для избежания дублирования)
    if logger.handlers:
        logger.handlers.clear()
    
    # 1. Файловый обработчик (подробные логи, уровень DEBUG)
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(file_level)
        
        # Подробный формат для файла: время, уровень, файл, строка, сообщение
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"⚠️ Не удалось создать файловый обработчик логов: {e}")
    
    # 2. Консольный обработчик (краткие логи, уровень INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    
    # Краткий формат для консоли
    console_format = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    logger.debug("Логгер успешно сконфигурирован")
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Получить логгер по имени или основной логгер приложения.
    
    Args:
        name: Имя логгера (если None, возвращает основной логгер)
    
    Returns:
        Логгер
    """
    if name:
        return logging.getLogger(f"lab4_async_processor.{name}")
    return logging.getLogger("lab4_async_processor")


# Создаем глобальный логгер при импорте
app_logger = setup_logging()