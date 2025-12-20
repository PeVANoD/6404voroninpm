"""
Модуль с классами для работы с изображениями животных.
"""
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import time
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
        #print(f"Метод {func.__name__} выполнен за {execution_time:.4f} секунд")
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
        self._image_processor = ImageProcessing()
    
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
    
    def _prepare_images_for_operation(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовить изображения для операций (приведение к одинаковому размеру и формату).
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h, w = min(h1, h2), min(w1, w2)
        
        img1_resized = img1[:h, :w]
        img2_resized = img2[:h, :w]
        
        # Приводим к одинаковому количеству каналов
        if len(img1_resized.shape) != len(img2_resized.shape):
            if len(img1_resized.shape) == 2:  # img1 - grayscale, img2 - color
                img1_resized = np.stack([img1_resized] * 3, axis=-1)
            else:  # img1 - color, img2 - grayscale
                img2_resized = np.stack([img2_resized] * 3, axis=-1)
        
        return img1_resized, img2_resized
    
    @abstractmethod
    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Абстрактный метод для преобразования в оттенки серого."""
        pass
    
    # Перегрузка операторов для работы с edges и другими изображениями
    def __add__(self, other) -> 'CatImage':
        """
        Перегрузка оператора + для сложения изображения 
        """
        if other == 'custom':
            edges = self.processed_edges_custom
            if edges is None:
                self.process_edges()
                edges = self._processed_edges_custom
            return self._add_with_edges(edges)
        elif isinstance(other, CatImage):
            return self._add_images(other)
        else:
            raise ValueError("Поддерживается только сложение с 'custom' или другим CatImage")
    
    def __sub__(self, other) -> 'CatImage':
        """
        Перегрузка оператора - для вычитания контуров из изображения 
        """
        if other == 'custom':
            edges = self.processed_edges_custom
            if edges is None:
                self.process_edges()
                edges = self._processed_edges_custom
            return self._subtract_edges(edges)
        elif isinstance(other, CatImage):
            return self._subtract_images(other)
        else:
            raise ValueError("Поддерживается только вычитание 'custom' или другого CatImage")
    
    @abstractmethod
    def _add_images(self, other: 'CatImage') -> 'CatImage':
        """Абстрактный метод для сложения двух изображений."""
        pass
    
    @abstractmethod
    def _subtract_images(self, other: 'CatImage') -> 'CatImage':
        """Абстрактный метод для вычитания двух изображений."""
        pass
    
    @abstractmethod
    def _add_with_edges(self, edges: np.ndarray) -> 'CatImage':
        """Абстрактный метод для сложения изображения с контурами."""
        pass
    
    @abstractmethod
    def _subtract_edges(self, edges: np.ndarray) -> 'CatImage':
        """Абстрактный метод для вычитания контуров из изображения."""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """Абстрактный метод для строкового представления изображения."""
        pass
    
    @timer_decorator
    def process_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Выполнить обнаружение контуров обоими методами.
        
        Returns:
            Кортеж (custom_edges, library_edges)
        """
        #print(f"Обработка контуров для породы: {self._breed}")
        
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
    
    def create_result_image(self, result_data: np.ndarray, operation: str = "", edges_type: str = "") -> 'CatImage':
        """Фабричный метод для создания результата того же типа."""
        if operation and edges_type:
            new_url = f"{self._image_url}_{operation}_{edges_type}_edges"
        elif operation:
            new_url = f"{self._image_url}_{operation}"
        else:
            new_url = f"{self._image_url}_result"
            
        new_breed = f"{self._breed}_{operation}" if operation else self._breed
        
        if isinstance(self, ColorCatImage):
            return ColorCatImage(result_data, new_url, new_breed)
        else:
            return GrayscaleCatImage(result_data, new_url, new_breed)


class ColorCatImage(CatImage):
    """
    Класс для работы с цветными изображениями животных.
    """
    
    def __init__(self, image_data: np.ndarray, image_url: str, breed: str):
        super().__init__(image_data, image_url, breed)
    
    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Преобразование RGB изображения в оттенки серого."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray, _ = self._image_processor.rgb_to_grayscale(image)
            return gray.astype(np.uint8)
        return image
    
    def _add_images(self, other: CatImage) -> 'ColorCatImage':
        """Сложение двух цветных изображений."""
        img1, img2 = self._prepare_images_for_operation(
            self._image_data, other.image_data
        )
        result = np.clip(img1.astype(np.int32) + img2.astype(np.int32), 0, 255)
        return self.create_result_image(result.astype(np.uint8), "added_with", f"{other.breed}")
    
    def _subtract_images(self, other: CatImage) -> 'ColorCatImage':
        """Вычитание двух цветных изображений."""
        img1, img2 = self._prepare_images_for_operation(
            self._image_data, other.image_data
        )
        result = np.clip(img1.astype(np.int32) - img2.astype(np.int32), 0, 255)
        return self.create_result_image(result.astype(np.uint8), "subtracted_with", f"{other.breed}")
    
    def _add_with_edges(self, edges: np.ndarray) -> 'ColorCatImage':
        """Сложение цветного изображения с его контурами."""
        original, edges_prepared = self._prepare_images_for_operation(
            self._image_data, edges
        )
        result = np.clip(original.astype(np.int32) + edges_prepared.astype(np.int32), 0, 255)
        
        # Определяем тип edges для названия
        edges_type = 'custom' if np.array_equal(edges, self._processed_edges_custom) else 'library'
        return self.create_result_image(result.astype(np.uint8), 'add', edges_type)
    
    def _subtract_edges(self, edges: np.ndarray) -> 'ColorCatImage':
        """Вычитание контуров из цветного изображения."""
        original, edges_prepared = self._prepare_images_for_operation(
            self._image_data, edges
        )
        result = np.clip(original.astype(np.int32) - edges_prepared.astype(np.int32), 0, 255)
        
        # Определяем тип edges для названия
        edges_type = 'custom' if np.array_equal(edges, self._processed_edges_custom) else 'library'
        return self.create_result_image(result.astype(np.uint8), 'subtract', edges_type)
    
    def __str__(self) -> str:
        """Строковое представление цветного изображения."""
        return (f"ColorCatImage(breed={self._breed}, "
                f"shape={self._image_data.shape} (color), "
                f"url={self._image_url})")
    
    def _custom_edge_detection(self) -> np.ndarray:
        """Пользовательское обнаружение контуров."""
        #print("Пользовательское обнаружение контуров (метод из lab1)...")
        edges, execution_time = self._image_processor.edge_detection(self._image_data)
        #print(f"Пользовательские контуры найдены за {execution_time:.4f} секунд")
        return edges
    
    def _library_edge_detection(self) -> np.ndarray:
        """Библиотечное обнаружение контуров с использованием OpenCV Canny."""
        #print("Библиотечное обнаружение контуров (OpenCV Canny)...")
        gray = self._rgb_to_grayscale(self._image_data)
        edges = cv2.Canny(gray, 50, 150)
        return edges


class GrayscaleCatImage(CatImage):
    """
    Класс для работы с чёрно-белыми изображениями животных.
    """
    
    def __init__(self, image_data: np.ndarray, image_url: str, breed: str):
        super().__init__(image_data, image_url, breed)
    
    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Для ч/б изображений преобразование не требуется."""
        return image
    
    def _add_images(self, other: CatImage) -> 'GrayscaleCatImage':
        """Сложение двух ч/б изображений."""
        img1, img2 = self._prepare_images_for_operation(
            self._image_data, other.image_data
        )
        result = np.clip(img1.astype(np.int32) + img2.astype(np.int32), 0, 255)
        return self.create_result_image(result.astype(np.uint8), "added_with", f"{other.breed}")
    
    def _subtract_images(self, other: CatImage) -> 'GrayscaleCatImage':
        """Вычитание двух ч/б изображений."""
        img1, img2 = self._prepare_images_for_operation(
            self._image_data, other.image_data
        )
        result = np.clip(img1.astype(np.int32) - img2.astype(np.int32), 0, 255)
        return self.create_result_image(result.astype(np.uint8), "subtracted_with", f"{other.breed}")
    
    def _add_with_edges(self, edges: np.ndarray) -> 'GrayscaleCatImage':
        """Сложение ч/б изображения с его контурами."""
        original, edges_prepared = self._prepare_images_for_operation(
            self._image_data, edges
        )
        result = np.clip(original.astype(np.int32) + edges_prepared.astype(np.int32), 0, 255)
        
        # Определяем тип edges для названия
        edges_type = 'custom' if np.array_equal(edges, self._processed_edges_custom) else 'library'
        return self.create_result_image(result.astype(np.uint8), 'add', edges_type)
    
    def _subtract_edges(self, edges: np.ndarray) -> 'GrayscaleCatImage':
        """Вычитание контуров из ч/б изображения."""
        original, edges_prepared = self._prepare_images_for_operation(
            self._image_data, edges
        )
        result = np.clip(original.astype(np.int32) - edges_prepared.astype(np.int32), 0, 255)
        
        # Определяем тип edges для названия
        edges_type = 'custom' if np.array_equal(edges, self._processed_edges_custom) else 'library'
        return self.create_result_image(result.astype(np.uint8), 'subtract', edges_type)
    
    def __str__(self) -> str:
        """Строковое представление ч/б изображения."""
        return (f"GrayscaleCatImage(breed={self._breed}, "
                f"shape={self._image_data.shape} (grayscale), "
                f"url={self._image_url})")
    
    def _custom_edge_detection(self) -> np.ndarray:
        """Пользовательское обнаружение контуров для ч/б изображений."""
        print("Пользовательское обнаружение контуров для ч/б изображения...")
        edges, execution_time = self._image_processor.edge_detection(self._image_data)
        return edges
    
    def _library_edge_detection(self) -> np.ndarray:
        """Библиотечное обнаружение контуров для ч/б изображений."""
        print("Библиотечное обнаружение контуров для ч/б изображения...")
        edges = cv2.Canny(self._image_data, 50, 150)
        return edges


def create_cat_image(image_data: np.ndarray, image_url: str, breed: str) -> CatImage:
    """
    Фабричный метод для создания подходящего типа изображения.
    
    Args:
        image_data: Данные изображения в виде numpy массива
        image_url: URL исходного изображения
        breed: Порода животного
        
    Returns:
        Объект ColorCatImage или GrayscaleCatImage в зависимости от типа изображения
    """
    if len(image_data.shape) == 2:
        # 2D массив - чёрно-белое изображение
        #print(f"Создаём GrayscaleCatImage для {breed}")
        return GrayscaleCatImage(image_data, image_url, breed)
    elif len(image_data.shape) == 3 and image_data.shape[2] == 1:
        # 3D массив с одним каналом - чёрно-белое
        #print(f"Создаём GrayscaleCatImage для {breed}")
        return GrayscaleCatImage(image_data.squeeze(), image_url, breed)
    else:
        # 3D массив с 3 или 4 каналами - цветное
        #print(f"Создаём ColorCatImage для {breed}")
        return ColorCatImage(image_data, image_url, breed)