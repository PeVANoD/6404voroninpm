"""
Тесты для классов CatImage.
"""

import unittest
import numpy as np
import cv2
import sys
import os

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cat_image import ColorCatImage, GrayscaleCatImage, create_cat_image


class TestCatImage(unittest.TestCase):
    """
    Тесты для функционала CatImage.
    """
    
    def setUp(self):
        """Подготовка тестовых данных."""
        # Создаем тестовое цветное изображение
        self.color_image_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.color_image = ColorCatImage(
            self.color_image_data, 
            "http://test.com/color.jpg", 
            "test_breed_color"
        )
        
        # Создаем тестовое ч/б изображение
        self.gray_image_data = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        self.gray_image = GrayscaleCatImage(
            self.gray_image_data,
            "http://test.com/gray.jpg",
            "test_breed_gray"
        )
    
    def test_color_image_creation(self):
        """Тест создания цветного изображения."""
        self.assertEqual(self.color_image.breed, "test_breed_color")
        self.assertEqual(self.color_image.image_data.shape, (100, 100, 3))
        self.assertTrue(isinstance(self.color_image, ColorCatImage))
    
    def test_grayscale_image_creation(self):
        """Тест создания ч/б изображения."""
        self.assertEqual(self.gray_image.breed, "test_breed_gray")
        self.assertEqual(self.gray_image.image_data.shape, (100, 100))
        self.assertTrue(isinstance(self.gray_image, GrayscaleCatImage))
    
    def test_factory_method(self):
        """Тест фабричного метода create_cat_image."""
        # Цветное изображение
        color_img = create_cat_image(
            np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8),
            "http://test.com/factory_color.jpg",
            "factory_color"
        )
        self.assertTrue(isinstance(color_img, ColorCatImage))
        
        # Ч/б изображение
        gray_img = create_cat_image(
            np.random.randint(0, 256, (50, 50), dtype=np.uint8),
            "http://test.com/factory_gray.jpg",
            "factory_gray"
        )
        self.assertTrue(isinstance(gray_img, GrayscaleCatImage))
    
    def test_rgb_to_grayscale_conversion(self):
        """Тест преобразования RGB в оттенки серого."""
        gray = self.color_image._rgb_to_grayscale(self.color_image_data)
        
        # Проверяем размерность
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape, (100, 100))
        
        # Проверяем диапазон значений
        self.assertTrue(np.all(gray >= 0))
        self.assertTrue(np.all(gray <= 255))
    
    def test_edge_detection(self):
        """Тест обнаружения контуров."""
        # Для цветного изображения
        custom_edges, library_edges = self.color_image.process_edges()
        
        self.assertIsNotNone(custom_edges)
        self.assertIsNotNone(library_edges)
        self.assertEqual(custom_edges.shape, (100, 100))
        self.assertEqual(library_edges.shape, (100, 100))
    
    def test_image_addition(self):
        """Тест сложения изображений."""
        # Создаем два тестовых изображения
        img1_data = np.ones((50, 50, 3), dtype=np.uint8) * 100
        img2_data = np.ones((50, 50, 3), dtype=np.uint8) * 50
        
        img1 = ColorCatImage(img1_data, "url1", "breed1")
        img2 = ColorCatImage(img2_data, "url2", "breed2")
        
        # Складываем
        result = img1 + img2
        
        # Проверяем результат
        self.assertTrue(isinstance(result, ColorCatImage))
        expected_result = np.clip(img1_data.astype(int) + img2_data.astype(int), 0, 255)
        self.assertTrue(np.array_equal(result.image_data, expected_result.astype(np.uint8)))
    
    def test_image_subtraction(self):
        """Тест вычитания изображений."""
        img1_data = np.ones((50, 50, 3), dtype=np.uint8) * 200
        img2_data = np.ones((50, 50, 3), dtype=np.uint8) * 100
        
        img1 = ColorCatImage(img1_data, "url1", "breed1")
        img2 = ColorCatImage(img2_data, "url2", "breed2")
        
        # Вычитаем
        result = img1 - img2
        
        # Проверяем результат
        self.assertTrue(isinstance(result, ColorCatImage))
        expected_result = np.clip(img1_data.astype(int) - img2_data.astype(int), 0, 255)
        self.assertTrue(np.array_equal(result.image_data, expected_result.astype(np.uint8)))
    
    def test_str_representation(self):
        """Тест строкового представления."""
        str_repr = str(self.color_image)
        self.assertIn("ColorCatImage", str_repr)
        self.assertIn("test_breed_color", str_repr)
        self.assertIn("shape=", str_repr)


class TestAsyncLoggingIntegration(unittest.TestCase):
    """
    Тесты интеграции с асинхронным логированием (для lab5).
    """
    
    def test_async_processor_v2_import(self):
        """Тест импорта AsyncCatImageProcessorV2."""
        try:
            from async_cat_image_processor_v2 import AsyncCatImageProcessorV2
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Не удалось импортировать AsyncCatImageProcessorV2: {e}")
    
    def test_async_logging_config_import(self):
        """Тест импорта async_logging_config."""
        try:
            from async_logging_config import setup_async_logging, get_async_logger
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Не удалось импортировать async_logging_config: {e}")


if __name__ == '__main__':
    unittest.main()