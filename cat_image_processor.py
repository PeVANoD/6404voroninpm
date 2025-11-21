"""
Модуль для работы с API и обработки изображений животных.
"""
import os
import requests
import numpy as np
from typing import List, Optional
import cv2
import time
import urllib3
from PIL import Image
import io

# Отключаем предупреждения о SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Импортируем конфигурацию и классы изображений
from config import API_KEY, BASE_URL, DEFAULT_LIMIT, DEFAULT_OUTPUT_DIR
from cat_image import CatImage, create_cat_image, timer_decorator


class CatImageProcessor:
    """
    Класс для обработки изображений животных через API.
    """
    
    def __init__(self, limit: int = DEFAULT_LIMIT, output_dir: str = DEFAULT_OUTPUT_DIR):
        """
        Инициализация процессора.
        
        Args:
            limit: Количество изображений для загрузки
            output_dir: Директория для сохранения результатов
        """
        self._limit = limit
        self._api_key = API_KEY
        self._base_url = BASE_URL
        self._downloaded_images: List[CatImage] = []
        self._output_dir = output_dir
        
        # Создаем сессию для повторного использования
        self._session = requests.Session()
        self._session.verify = False
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'x-api-key': self._api_key
        })
    
    @property
    def downloaded_images(self) -> List[CatImage]:
        """Получить список загруженных изображений."""
        return self._downloaded_images
    
    @timer_decorator
    def _create_output_directory(self) -> None:
        """Создать директорию для сохранения изображений."""
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)
            print(f"Создана директория: {self._output_dir}")
    
    @timer_decorator
    def _fetch_images_from_api(self) -> List[dict]:
        """
        Получить изображения из API.
        
        Returns:
            Список данных об изображениях
        """
        print(f"Запрос {self._limit} изображений из API...")
        
        params = {'limit': self._limit, 'has_breeds': 1}
        
        try:
            response = self._session.get(
                f"{self._base_url}/images/search",
                params=params,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Ошибка API: {response.status_code} - {response.text}")
            
            data = response.json()
            print(f"✓ Получено {len(data)} изображений из API")
            return data
            
        except Exception as e:
            print(f"✗ Ошибка при запросе к API: {e}")
            raise
    
    @timer_decorator
    def _fetch_single_image(self) -> Optional[dict]:
        """
        Получить одно изображение из API.
        
        Returns:
            Данные одного изображения или None в случае ошибки
        """
        try:
            params = {'limit': 1, 'has_breeds': 1}
            
            response = self._session.get(
                f"{self._base_url}/images/search",
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]
            
            return None
            
        except Exception as e:
            print(f"Ошибка при получении одного изображения: {e}")
            return None

    @timer_decorator
    def _download_image(self, url: str) -> Optional[np.ndarray]:
        """
        Скачать изображение по URL.
        """
        max_attempts = 3
        timeout = 15
        
        for attempt in range(max_attempts):
            try:
                print(f"Попытка {attempt + 1}/{max_attempts} загрузки...")
                
                response = self._session.get(url, timeout=timeout)
                response.raise_for_status()
                
                content = response.content
                
                if not content:
                    print("Получен пустой ответ")
                    continue
                
                # Декодируем изображение с помощью PIL
                pil_image = Image.open(io.BytesIO(content))
                
                # Конвертируем в RGB если нужно
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                elif pil_image.mode == 'P':
                    pil_image = pil_image.convert('RGB')
                
                # Преобразуем в numpy array
                image_array = np.array(pil_image)
                
                print(f"✓ Изображение загружено: {image_array.shape}")
                return image_array
                
            except Exception as e:
                print(f"Ошибка при загрузке (попытка {attempt + 1}): {e}")
        
        print("Все попытки загрузки завершились неудачно")
        return None

    @timer_decorator
    def _get_breed_name(self, image_data: dict) -> str:
        """
        Извлечь название породы из данных изображения.
        """
        breeds = image_data.get('breeds', [])
        if breeds and len(breeds) > 0:
            breed_name = breeds[0].get('name', 'unknown')
            # Заменяем проблемные символы в названии породы
            breed_name = breed_name.replace(' ', '_').replace('/', '_').replace('\\', '_').lower()
            return breed_name
        return 'unknown'
    
    @timer_decorator
    def _save_image(self, image: np.ndarray, filename: str) -> bool:
        """
        Сохранить изображение в файл.
        
        Returns:
            True если успешно, False если ошибка
        """
        try:
            filepath = os.path.join(self._output_dir, filename)
            
            # Для grayscale изображений используем соответствующий флаг
            if len(image.shape) == 2:
                success = cv2.imwrite(filepath, image)
            else:
                # Конвертируем RGB в BGR для OpenCV
                if image.shape[2] == 3:
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    success = cv2.imwrite(filepath, image_bgr)
                else:
                    success = cv2.imwrite(filepath, image)
            
            if success:
                print(f"✓ Изображение сохранено: {filename}")
                return True
            else:
                print(f"✗ Ошибка при сохранении: {filename}")
                return False
                
        except Exception as e:
            print(f"✗ Ошибка при сохранении изображения {filename}: {e}")
            return False
    
    @timer_decorator
    def download_images(self) -> None:
        """Загрузить изображения из API."""
        try:
            api_data = self._fetch_images_from_api()
            successful_downloads = 0
            max_attempts_per_image = 5  # Максимальное количество попыток для одного изображения
            
            for i, image_data in enumerate(api_data):
                print(f"\n{'='*50}")
                print(f"ЗАГРУЗКА ИЗОБРАЖЕНИЯ {i+1}/{len(api_data)}")
                print(f"{'='*50}")
                
                image_url = image_data['url']
                breed = self._get_breed_name(image_data)
                
                print(f"Порода: {breed}")
                print(f"URL: {image_url}")
                
                # Пытаемся скачать изображение подходящего размера
                image_array = None
                current_attempt_data = image_data
                current_url = image_url
                current_breed = breed
                
                for attempt in range(max_attempts_per_image):
                    print(f"\nПопытка {attempt + 1}/{max_attempts_per_image}")
                    
                    downloaded_image = self._download_image(current_url)
                    if downloaded_image is not None:
                        # Проверяем размер изображения
                        height, width = downloaded_image.shape[:2]
                        print(f"Размер изображения: {width}x{height}")
                        
                        if width <= 2000 and height <= 2000:
                            image_array = downloaded_image
                            print(f"✓ Размер изображения подходит: {width}x{height}")
                            break
                        else:
                            print(f"✗ Изображение слишком большое: {width}x{height} > 2000x2000")
                            
                            # Если есть еще попытки, получаем новое изображение
                            if attempt < max_attempts_per_image - 1:
                                print("Пробуем получить другое изображение...")
                                new_image_data = self._fetch_single_image()
                                if new_image_data:
                                    current_url = new_image_data['url']
                                    current_breed = self._get_breed_name(new_image_data)
                                    print(f"Новый URL: {current_url}")
                                    print(f"Новая порода: {current_breed}")
                                else:
                                    print("Не удалось получить новое изображение")
                    else:
                        print("Не удалось загрузить изображение")
                
                if image_array is not None:
                    # Используем фабричный метод для создания подходящего типа изображения
                    cat_image = create_cat_image(image_array, current_url, current_breed)
                    
                    self._downloaded_images.append(cat_image)
                    successful_downloads += 1
                    print(f"✓ УСПЕШНО ЗАГРУЖЕНО: {cat_image}")
                else:
                    print(f"✗ НЕ УДАЛОСЬ ЗАГРУЗИТЬ подходящее изображение {i+1}")
            
            print(f"\n{'='*50}")
            print(f"ИТОГ ЗАГРУЗКИ")
            print(f"{'='*50}")
            print(f"✓ Успешно: {successful_downloads}/{len(api_data)}")
            print(f"✗ Неудачно: {len(api_data) - successful_downloads}/{len(api_data)}")
            
        except Exception as e:
            print(f"✗ КРИТИЧЕСКАЯ ОШИБКА при загрузке изображений: {e}")
            raise
    
    @timer_decorator
    def process_images(self) -> None:
        """Обработать все загруженные изображения."""
        if not self._downloaded_images:
            print("Нет загруженных изображений. Загружаем...")
            self.download_images()
        
        if not self._downloaded_images:
            print("✗ Нет изображений для обработки")
            return
        
        self._create_output_directory()
        
        print(f"\nНачинаем обработку {len(self._downloaded_images)} изображений...")
        
        for i, cat_image in enumerate(self._downloaded_images):
            print(f"\n--- Обработка изображения {i+1}/{len(self._downloaded_images)} ---")
            print(f"Порода: {cat_image.breed}")
            print(f"Тип: {cat_image}")
            
            try:
                # Обработка контуров
                custom_edges, library_edges = cat_image.process_edges()
                
                # Сохранение изображений
                base_filename = f"{i+1:02d}_{cat_image.breed}"
                
                # Исходное изображение
                original_filename = f"{base_filename}_original.png"
                self._save_image(cat_image.image_data, original_filename)
                
                # Контуры пользовательским методом
                custom_filename = f"{base_filename}_custom_edges.png"
                self._save_image(custom_edges, custom_filename)
                
                # Контуры библиотечным методом
                library_filename = f"{base_filename}_library_edges.png"
                self._save_image(library_edges, library_filename)
                
                # Операции между оригиналом и его собственными контурами
                print("Выполняем операции между оригиналом и его контурами...")
                
                # Сложение оригинального изображения с его контурами
                added_custom_result = cat_image + 'custom'
                addition_custom_filename = f"{base_filename}_ORIGINAL_PLUS_CUSTOM.png"
                self._save_image(added_custom_result.image_data, addition_custom_filename)
                added_custom_result = cat_image + cat_image
                addition_custom_filename = f"{base_filename}_ORIGINAL_PLUS_ORIGINAL.png"
                self._save_image(added_custom_result.image_data, addition_custom_filename)
                
                # Вычитание контуров из оригинального изображения
                subtracted_custom_result = cat_image - 'custom'  # Перегруженный оператор -
                subtraction_custom_filename = f"{base_filename}_ORIGINAL_MINUS_CUSTOM.png"
                self._save_image(subtracted_custom_result.image_data, subtraction_custom_filename)
                
                print(f"✓ Завершена обработка изображения {i+1}")
                
            except Exception as e:
                print(f"✗ Ошибка при обработке изображения {i+1}: {e}")
                continue
    
    @timer_decorator
    def demonstrate_operations(self) -> None:
        """
        Продемонстрировать операции (упрощенная версия - только логирование).
        """
        print("\n--- Демонстрация операций ---")
        print("Все операции (сложение и вычитание оригинального изображения")
        print("с его контурами) уже выполнены в process_images()")
        print("Результаты сохранены в соответствующие файлы")