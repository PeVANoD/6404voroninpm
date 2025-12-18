"""
async_logging_config.py

Асинхронная конфигурация логирования для лабораторной работы №5.
Использует асинхронные обработчики для логирования в файл и консоль.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class AsyncFileHandler(logging.Handler):
    """
    Асинхронный обработчик логов для записи в файл.
    """
    
    def __init__(self, filename: str, mode: str = 'a', encoding: str = 'utf-8'):
        super().__init__()
        self.filename = filename
        self.mode = mode
        self.encoding = encoding
        
    def emit(self, record: logging.LogRecord) -> None:
        """Записываем лог в файл."""
        try:
            msg = self.format(record)
            with open(self.filename, self.mode, encoding=self.encoding) as f:
                f.write(msg + '\n')
        except Exception:
            self.handleError(record)


def setup_async_logging(
    log_file: str = "logs/async_app.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Настройка асинхронного логирования.
    
    Args:
        log_file: Путь к файлу логов
        console_level: Уровень логирования для консоли
        file_level: Уровень логирования для файла
    
    Returns:
        Сконфигурированный логгер
    """
    # Создаем директорию для логов, если её нет
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Создаем логгер
    logger = logging.getLogger("lab5_async_processor")
    logger.setLevel(logging.DEBUG)
    
    # Очищаем существующие обработчики
    if logger.handlers:
        logger.handlers.clear()
    
    # 1. Файловый обработчик
    file_handler = AsyncFileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    
    # Подробный формат для файла
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - '
        '[%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # 2. Консольный обработчик
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    
    # Краткий формат для консоли
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    logger.debug("Асинхронный логгер успешно сконфигурирован")
    
    return logger


def get_async_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Получить асинхронный логгер по имени.
    
    Args:
        name: Имя логгера (если None, возвращает основной логгер)
    
    Returns:
        Логгер
    """
    if name:
        return logging.getLogger(f"lab5_async_processor.{name}")
    return logging.getLogger("lab5_async_processor")