"""
async_logging_config.py

Асинхронная конфигурация логирования для лабораторной работы №5.
Читает настройки из JSON файла конфигурации.
"""

import logging
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
import os


class AsyncFileHandler(logging.Handler):
    """
    Простой асинхронный обработчик логов для записи в файл.
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


def load_logging_config(config_path: str = "logging_config.json") -> Optional[Dict[str, Any]]:
    """
    Загружает конфигурацию логирования из JSON файла.
    
    Args:
        config_path: Путь к JSON файлу конфигурации
    
    Returns:
        Словарь с конфигурацией или None если файл не найден
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config.get("logging")
        else:
            # Создаем конфиг по умолчанию
            default_config = create_default_config()
            save_logging_config(default_config, config_path)
            #print(f"📝 Создан файл конфигурации по умолчанию: {config_path}")
            return default_config.get("logging")
    except json.JSONDecodeError as e:
        print(f"❌ Ошибка чтения JSON конфигурации: {e}")
        return None
    except Exception as e:
        print(f"❌ Ошибка загрузки конфигурации: {e}")
        return None


def create_default_config() -> Dict[str, Any]:
    """Создает конфигурацию логирования по умолчанию."""
    return {
        "logging": {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
                "simple": {
                    "format": "%(asctime)s - %(levelname)s - %(message)s",
                    "datefmt": "%H:%M:%S"
                }
            },
            "handlers": {
                "async_file": {
                    "class": "AsyncFileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": "logs/async_app.log",
                    "mode": "a",
                    "encoding": "utf-8"
                },
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "simple",
                    "stream": "ext://sys.stdout"
                }
            },
            "loggers": {
                "lab5_async_processor": {
                    "level": "DEBUG",
                    "handlers": ["async_file", "console"],
                    "propagate": False
                }
            }
        }
    }


def save_logging_config(config: Dict[str, Any], config_path: str = "logging_config.json") -> None:
    """Сохраняет конфигурацию в JSON файл."""
    try:
        # Создаем директорию, если нужно
        config_dir = os.path.dirname(config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"❌ Ошибка сохранения конфигурации: {e}")


def setup_async_logging(
    config_file: Optional[str] = None,
    logger_name: str = "lab5_async_processor"
) -> logging.Logger:
    """
    Настройка асинхронного логирования из JSON конфига.
    
    Args:
        config_file: Путь к JSON файлу конфигурации
        logger_name: Имя логгера
    
    Returns:
        Сконфигурированный логгер
    """
    # Определяем путь к конфигу
    config_path = config_file or "logging_config.json"
    
    # Загружаем конфигурацию
    config = load_logging_config(config_path)
    if not config:
        print("⚠️ Используются настройки логирования по умолчанию")
        config = create_default_config().get("logging")
    
    # Создаем логгер
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Очищаем существующие обработчики
    if logger.handlers:
        logger.handlers.clear()
    
    # Настраиваем обработчики из конфига
    handlers_config = config.get("handlers", {})
    
    for handler_name, handler_cfg in handlers_config.items():
        # Пропускаем неактивные обработчики
        if handler_cfg.get("disabled", False):
            continue
            
        handler_class = handler_cfg.get("class")
        handler_level = getattr(logging, handler_cfg.get("level", "DEBUG"))
        formatter_name = handler_cfg.get("formatter", "simple")
        
        # Создаем обработчик
        if handler_class == "AsyncFileHandler":
            # Создаем директорию для логов
            log_file = handler_cfg.get("filename", "logs/async_app.log")
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            handler = AsyncFileHandler(
                filename=log_file,
                mode=handler_cfg.get("mode", "a"),
                encoding=handler_cfg.get("encoding", "utf-8")
            )
        elif handler_class == "logging.StreamHandler":
            stream = handler_cfg.get("stream", "ext://sys.stdout")
            if stream == "ext://sys.stdout":
                handler = logging.StreamHandler(sys.stdout)
            elif stream == "ext://sys.stderr":
                handler = logging.StreamHandler(sys.stderr)
            else:
                handler = logging.StreamHandler(sys.stdout)
        else:
            # Пропускаем неизвестные обработчики
            continue
        
        handler.setLevel(handler_level)
        
        # Настраиваем форматер
        formatters_config = config.get("formatters", {})
        if formatter_name in formatters_config:
            fmt_cfg = formatters_config[formatter_name]
            formatter = logging.Formatter(
                fmt=fmt_cfg.get("format", "%(asctime)s - %(levelname)s - %(message)s"),
                datefmt=fmt_cfg.get("datefmt", "%H:%M:%S")
            )
            handler.setFormatter(formatter)
        
        logger.addHandler(handler)
    
    # Настраиваем уровень логгера
    loggers_config = config.get("loggers", {})
    if logger_name in loggers_config:
        logger_level = loggers_config[logger_name].get("level", "DEBUG")
        logger.setLevel(getattr(logging, logger_level))
    
    logger.debug(f"Логгер '{logger_name}' успешно сконфигурирован из файла {config_path}")
    
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