"""
Тесты для асинхронного логирования.
"""

import unittest
import logging
import tempfile
import os
import sys

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from async_logging_config import setup_async_logging, get_async_logger


class TestAsyncLogging(unittest.TestCase):
    """
    Тесты для функционала асинхронного логирования.
    """
    
    def setUp(self):
        """Подготовка тестовых данных."""
        # Создаем временный файл для логов
        self.temp_log_file = tempfile.NamedTemporaryFile(
            suffix='.log', 
            delete=False,
            mode='w',
            encoding='utf-8'
        )
        self.temp_log_file.close()
        self.log_file_path = self.temp_log_file.name
    
    def tearDown(self):
        """Очистка после тестов."""
        # Удаляем временный файл
        if os.path.exists(self.log_file_path):
            os.unlink(self.log_file_path)
    
    def test_setup_async_logging(self):
        """Тест настройки асинхронного логирования."""
        logger = setup_async_logging(
            log_file=self.log_file_path,
            console_level=logging.WARNING,
            file_level=logging.DEBUG
        )
        
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "lab5_async_processor")
        self.assertEqual(logger.level, logging.DEBUG)
        
        # Проверяем наличие обработчиков
        self.assertEqual(len(logger.handlers), 2)
    
    def test_get_async_logger(self):
        """Тест получения именованного логгера."""
        main_logger = get_async_logger()
        self.assertEqual(main_logger.name, "lab5_async_processor")
        
        child_logger = get_async_logger("test.module")
        self.assertEqual(child_logger.name, "lab5_async_processor.test.module")
    
    def test_logger_levels(self):
        """Тест различных уровней логирования."""
        logger = setup_async_logging(
            log_file=self.log_file_path,
            console_level=logging.WARNING,  # В консоль только WARNING и выше
            file_level=logging.DEBUG        # В файл все сообщения
        )
        
        # Проверяем, что логгер имеет нужный уровень
        self.assertEqual(logger.level, logging.DEBUG)
        
        # Создаем тестовое сообщение для каждого уровня
        test_messages = {
            logging.DEBUG: "Debug message",
            logging.INFO: "Info message",
            logging.WARNING: "Warning message",
            logging.ERROR: "Error message",
            logging.CRITICAL: "Critical message"
        }
        
        for level, message in test_messages.items():
            logger.log(level, message)
        
        # Ждем записи
        import time
        time.sleep(0.1)
        
        # Читаем файл и проверяем, что сообщения записаны
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        for message in test_messages.values():
            self.assertIn(message, content)
    
    def test_async_logging_integration(self):
        """Тест интеграции асинхронного логирования."""
        # Настраиваем логирование
        logger = setup_async_logging(log_file=self.log_file_path)
        
        # Логируем разные уровни
        logger.debug("Отладочное сообщение")
        logger.info("Информационное сообщение")
        logger.warning("Предупреждение")
        logger.error("Ошибка")
        logger.critical("Критическая ошибка")
        
        # Проверяем, что сообщения записаны в файл
        import time
        time.sleep(0.1)
        
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        self.assertIn("Отладочное сообщение", content)
        self.assertIn("Информационное сообщение", content)
        self.assertIn("Предупреждение", content)
        self.assertIn("Ошибка", content)
        self.assertIn("Критическая ошибка", content)


if __name__ == '__main__':
    unittest.main()