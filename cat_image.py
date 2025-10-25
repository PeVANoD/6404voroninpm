"""
Модуль с классами для работы с изображениями животных.
"""
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union
import time
from PIL import Image, ImageFilter
import cv2

# Импортируем ImageProcessing из первой лабораторной
from implementation import ImageProcessing


def timer_decorator(func):
    """
    Декоратор для измерения времени выполнения методов.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Метод {func.__name__} выполнен за {execution_time:.4f} секунд")
        return result
    return wrapper


class CatImage(ABC):
    """
    Абстрактный базовый класс для работы с изображениями животных.
    """
    
    def __init__(self, image_data: np.ndarray, image_url: str, breed: str):
        """
        Инициализация изображения.
        
        Args:
            image_data: Данные изображения в виде numpy массива
            image_url: URL исходного изображения
            breed: Порода животного
        """
        self._image_data = image_data
        self._image_url = image_url
        self._breed = breed
        self._processed_edges_custom = None
        self._processed_edges_library = None
        self._image_processor = ImageProcessing()  # Процессор из первой лабы
    
    @property
    def image_data(self) -> np.ndarray:
        """Получить данные изображения."""
        return self._image_data
    
    @property
    def image_url(self) -> str:
        """Получить URL изображения."""
        return self._image_url
    
    @property
    def breed(self) -> str:
        """Получить породу животного."""
        return self._breed
    
    @property
    def processed_edges_custom(self) -> Optional[np.ndarray]:
        """Получить контуры, обработанные пользовательским методом."""
        return self._processed_edges_custom
    
    @property
    def processed_edges_library(self) -> Optional[np.ndarray]:
        """Получить контуры, обработанные библиотечным методом."""
        return self._processed_edges_library
    
    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Преобразование RGB изображения в оттенки серого.
        """
        if len(image.shape) == 3:
            # Используем метод из первой лабораторной
            gray, _ = self._image_processor.rgb_to_grayscale(image)
            return gray.astype(np.uint8)
        else:
            return image
    
    @timer_decorator
    def process_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Выполнить обнаружение контуров обоими методами.
        
        Returns:
            Кортеж (custom_edges, library_edges)
        """
        print(f"Обработка контуров для породы: {self._breed}")
        
        self._processed_edges_custom = self._custom_edge_detection()
        self._processed_edges_library = self._library_edge_detection()
        
        return self._processed_edges_custom, self._processed_edges_library
    
    @abstractmethod
    def _custom_edge_detection(self) -> np.ndarray:
        """Абстрактный метод для пользовательского обнаружения контуров."""
        pass
    
    @abstractmethod
    def _library_edge_detection(self) -> np.ndarray:
        """Абстрактный метод для библиотечного обнаружения контуров."""
        pass
    
    def _add_images(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Покомпонентное сложение двух изображений.
        """
        # Приводим к одинаковому размеру
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h, w = min(h1, h2), min(w1, w2)
        
        img1_resized = img1[:h, :w]
        img2_resized = img2[:h, :w]
        
        # Сложение с ограничением до 255
        result = np.clip(img1_resized.astype(np.int32) + img2_resized.astype(np.int32), 0, 255)
        return result.astype(np.uint8)
    
    def _subtract_images(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Покомпонентное вычитание двух изображений.
        """
        # Приводим к одинаковому размеру
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h, w = min(h1, h2), min(w1, w2)
        
        img1_resized = img1[:h, :w]
        img2_resized = img2[:h, :w]
        
        # Вычитание с ограничением до 0
        result = np.clip(img1_resized.astype(np.int32) - img2_resized.astype(np.int32), 0, 255)
        return result.astype(np.uint8)
    
    def __add__(self, other: 'CatImage') -> 'CatImage':
        """
        Сложение двух изображений (покомпонентное).
        """
        if not isinstance(other, CatImage):
            raise TypeError("Можно складывать только объекты CatImage")
        
        result_data = self._add_images(self._image_data, other.image_data)
        return self.__class__(result_data, f"combined_{self._breed}", self._breed)
    
    def __sub__(self, other: 'CatImage') -> 'CatImage':
        """
        Вычитание двух изображений (покомпонентное).
        """
        if not isinstance(other, CatImage):
            raise TypeError("Можно вычитать только объекты CatImage")
        
        result_data = self._subtract_images(self._image_data, other.image_data)
        return self.__class__(result_data, f"subtracted_{self._breed}", self._breed)
    
    def __str__(self) -> str:
        """
        Строковое представление изображения.
        """
        shape_str = f"shape={self._image_data.shape}"
        if hasattr(self, '_is_grayscale') and self._is_grayscale:
            shape_str += " (grayscale)"
        else:
            shape_str += " (color)"
            
        return (f"{self.__class__.__name__}(breed={self._breed}, "
                f"{shape_str}, "
                f"url={self._image_url})")


class ColorCatImage(CatImage):
    """
    Класс для работы с цветными изображениями животных.
    """
    
    def __init__(self, image_data: np.ndarray, image_url: str, breed: str):
        super().__init__(image_data, image_url, breed)
        self._is_grayscale = False
    
    def _custom_edge_detection(self) -> np.ndarray:
        """
        Пользовательское обнаружение контуров с использованием метода из первой лабораторной.
        
        Returns:
            Изображение с выделенными контурами
        """
        print("Пользовательское обнаружение контуров (метод из lab1)...")
        
        # Используем метод edge_detection из первой лабораторной
        edges, execution_time = self._image_processor.edge_detection(self._image_data)
        print(f"Пользовательские контуры найдены за {execution_time:.4f} секунд")
        
        return edges
    
    def _library_edge_detection(self) -> np.ndarray:
        """
        Библиотечное обнаружение контуров с использованием OpenCV Canny.
        
        Returns:
            Изображение с выделенными контурами
        """
        print("🔍 Библиотечное обнаружение контуров (OpenCV Canny)...")
        
        # Преобразуем в градации серого
        gray = self._rgb_to_grayscale(self._image_data)
        
        # Используем детектор границ Canny из OpenCV
        edges = cv2.Canny(gray, 50, 150)
        
        return edges
