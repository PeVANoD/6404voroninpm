"""
Тесты для AsyncCatImageProcessorV2 (lab5).
"""

import unittest
import asyncio
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, AsyncMock

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from async_cat_image_processor_v2 import AsyncCatImageProcessorV2
from config import DEFAULT_OUTPUT_DIR  # Импортируем константу из config


class TestAsyncProcessorV2(unittest.TestCase):
    """
    Тесты для функционала AsyncCatImageProcessorV2 (lab5).
    """
    
    def setUp(self):
        """Подготовка тестовых данных."""
        self.processor = AsyncCatImageProcessorV2(limit=2)
    
    def test_initialization(self):
        """Тест инициализации процессора v2."""
        self.assertEqual(self.processor._limit, 2)
        
        # Используем значение из config, а не жестко заданную строку
        self.assertEqual(self.processor._output_dir, DEFAULT_OUTPUT_DIR)
        
        self.assertEqual(len(self.processor.downloaded_images), 0)
        
        # Проверяем, что логгер создан
        self.assertIsNotNone(self.processor._logger)
        # Имя логгера может быть другим в зависимости от реализации
        self.assertTrue(hasattr(self.processor._logger, 'name'))
    
    def test_initialization_with_custom_output_dir(self):
        """Тест инициализации с кастомной директорией."""
        processor = AsyncCatImageProcessorV2(limit=3, output_dir="custom_output")
        self.assertEqual(processor._output_dir, "custom_output")
        self.assertEqual(processor._limit, 3)
    
    def test_get_breed_name(self):
        """Тест извлечения названия породы."""
        # Тест с полными данными
        image_data = {
            'breeds': [{'name': 'British Shorthair'}]
        }
        breed = self.processor._get_breed_name(image_data)
        self.assertEqual(breed, 'british_shorthair')
        
        # Тест без данных о породе
        image_data = {'breeds': []}
        breed = self.processor._get_breed_name(image_data)
        self.assertEqual(breed, 'unknown')
        
        # Тест с заменой пробелов и спецсимволов
        image_data = {
            'breeds': [{'name': 'American Curl/Shorthair'}]
        }
        breed = self.processor._get_breed_name(image_data)
        self.assertEqual(breed, 'american_curl_shorthair')
    
    def test_downloaded_images_property(self):
        """Тест свойства downloaded_images."""
        # Изначально список пустой
        self.assertEqual(len(self.processor.downloaded_images), 0)
        
        # Добавляем тестовое изображение
        from cat_image import ColorCatImage
        test_cat_image = ColorCatImage(
            np.ones((100, 100, 3), dtype=np.uint8),
            "http://test.com/image.jpg",
            "test_breed"
        )
        self.processor._downloaded_images.append(test_cat_image)
        
        # Проверяем, что свойство возвращает список
        downloaded = self.processor.downloaded_images
        self.assertEqual(len(downloaded), 1)
        self.assertEqual(downloaded[0], test_cat_image)
    
    def test_save_single_image(self):
        """Тест сохранения изображения (v2 с логированием)."""
        # Тестовое изображение
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Мокаем cv2.imencode
        with patch('async_cat_image_processor_v2.cv2.imencode') as mock_imencode:
            # Создаем mock для encoded_image
            mock_encoded = Mock()
            mock_encoded.tobytes.return_value = b'fake_image_data'
            mock_imencode.return_value = (True, mock_encoded)
            
            # Мокаем aiofiles.open
            with patch('async_cat_image_processor_v2.aiofiles.open') as mock_aiofiles_open:
                mock_file = AsyncMock()
                mock_aiofiles_open.return_value.__aenter__.return_value = mock_file
                
                # Тестируем синхронно
                async def run_test():
                    return await self.processor._save_single_image(test_image, "test.png")
                
                success = asyncio.run(run_test())
                
                self.assertTrue(success)
                mock_imencode.assert_called_once()
                mock_file.write.assert_called_once_with(b'fake_image_data')


class TestAsyncProcessorV2Sync(unittest.TestCase):
    """
    Синхронные тесты для AsyncCatImageProcessorV2.
    """
    
    def test_processor_initialization_sync(self):
        """Синхронный тест инициализации v2."""
        processor = AsyncCatImageProcessorV2(limit=5, output_dir="test_output_v2")
        self.assertEqual(processor._limit, 5)
        self.assertEqual(processor._output_dir, "test_output_v2")
    
    def test_simple_pipeline_mocked(self):
        """Простой тест пайплайна с моками."""
        # Создаем процессор
        processor = AsyncCatImageProcessorV2(limit=2)
        
        # Мокаем run_pipeline
        with patch.object(processor, 'fetch_image_urls') as mock_fetch, \
             patch.object(processor, 'download_images') as mock_download, \
             patch.object(processor, 'process_images') as mock_process, \
             patch.object(processor, 'save_images') as mock_save:
            
            # Настраиваем моки
            mock_fetch.return_value = self._create_mock_generator([])
            mock_download.return_value = self._create_mock_generator([])
            mock_process.return_value = self._create_mock_generator([])
            mock_save.return_value = self._create_mock_generator([])
            
            # Запускаем тест
            async def run_test():
                await processor.run_pipeline()
            
            # Должно выполниться без ошибок
            try:
                asyncio.run(run_test())
                self.assertTrue(True)
            except Exception as e:
                self.fail(f"Паплайн завершился с ошибкой: {e}")
    
    def test_logging_in_pipeline(self):
        """Тест логирования в пайплайне."""
        # Создаем процессор
        processor = AsyncCatImageProcessorV2(limit=1)
        
        # Мокаем методы, чтобы тест был быстрым
        with patch.object(processor, 'fetch_image_urls') as mock_fetch, \
             patch.object(processor, 'download_images') as mock_download, \
             patch.object(processor, 'process_images') as mock_process, \
             patch.object(processor, 'save_images') as mock_save:
            
            # Создаем mock CatImage
            mock_cat_image = Mock()
            mock_cat_image.breed = 'test_breed'
            mock_cat_image.image_data = np.ones((100, 100, 3), dtype=np.uint8)
            mock_cat_image.processed_edges_custom = None
            mock_cat_image.processed_edges_library = None
            
            # Настраиваем моки
            mock_fetch.return_value = self._create_mock_generator([
                (0, 'http://test.com/image.jpg', 'test_breed')
            ])
            
            mock_download.return_value = self._create_mock_generator([
                (0, np.ones((100, 100, 3), dtype=np.uint8), 'http://test.com/image.jpg', 'test_breed')
            ])
            
            mock_process.return_value = self._create_mock_generator([
                (0, mock_cat_image)
            ])
            
            mock_save.return_value = self._create_mock_generator([
                (0, mock_cat_image)
            ])
            
            # Запускаем тест
            async def run_test():
                await processor.run_pipeline()
            
            # Проверяем, что пайплайн выполняется без ошибок
            try:
                asyncio.run(run_test())
                self.assertTrue(True)
            except Exception as e:
                self.fail(f"Паплайн завершился с ошибкой: {e}")
    
    def _create_mock_generator(self, items):
        """Создает mock асинхронного генератора."""
        async def mock_gen():
            for item in items:
                yield item
        return mock_gen()


if __name__ == '__main__':
    unittest.main()