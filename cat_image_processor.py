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

# Отключаем предупреждения о SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Импортируем конфигурацию
from config import API_KEY, BASE_URL, DEFAULT_LIMIT, DEFAULT_OUTPUT_DIR
from cat_image import ColorCatImage, timer_decorator


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
        self._downloaded_images: List = []
        self._output_dir = output_dir
        
        print(f"API ключ загружен")
    
    @property
    def downloaded_images(self) -> List:
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
        
        headers = {'x-api-key': self._api_key}
        params = {'limit': self._limit, 'has_breeds': 1}
        
        try:
            session = requests.Session()
            session.verify = False
            
            response = session.get(
                f"{self._base_url}/images/search",
                headers=headers,
                params=params,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Ошибка API: {response.status_code} - {response.text}")
            
            data = response.json()
            print(f"O Получено {len(data)} изображений из API")
            return data
            
        except Exception as e:
            print(f"X Ошибка при запросе к API: {e}")
            raise
    
    @timer_decorator
    def _download_image(self, url: str) -> Optional[np.ndarray]:
        """
        Скачать изображение.
        """
        max_attempts = 2
        timeout = 10
        
        print(f"Ссылка для скачивания: {url}")
        
        for attempt in range(max_attempts):
            try:
                print(f"\n! Попытка {attempt + 1}/{max_attempts} загрузки...")
                
                session = requests.Session()
                session.verify = False
                session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
                })
                
                print("Отправляем GET запрос...")
                start_time = time.time()
                response = session.get(url, timeout=timeout)
                download_time = time.time() - start_time
                
                print(f"P Ответ получен за {download_time:.2f} секунд")
                print(f"Статус код: {response.status_code}")
                
                response.raise_for_status()
                
                content = response.content
                
                if not content:
                    print("X Получен пустой ответ")
                    continue
                
                print(f"Размер данных: {len(content)} байт ({len(content)/1024:.1f} KB)")
                
                try:
                    # Декодируем изображение с помощью PIL
                    from PIL import Image
                    import io
                    
                    print("Декодируем изображение с помощью PIL...")
                    
                    # Открываем изображение из байтов
                    pil_image = Image.open(io.BytesIO(content))
                    
                    # Преобразуем PIL Image в numpy array
                    if pil_image.mode == 'RGB':
                        image_array = np.array(pil_image)
                        print(f"O RGB изображение декодировано")
                    elif pil_image.mode == 'RGBA':
                        # Конвертируем RGBA в RGB
                        rgb_image = pil_image.convert('RGB')
                        image_array = np.array(rgb_image)
                        print(f"O RGBA изображение конвертировано в RGB")
                    
                    print(f"Размер изображения: {image_array.shape}")
                    print(f"Тип данных: {image_array.dtype}")
                    print(f"Диапазон значений: [{image_array.min()}, {image_array.max()}]")
                    
                    return image_array
                    
                except Exception as pil_error:
                    print(f"X Ошибка при декодировании PIL: {pil_error}")
            except Exception as e:
                print(f"X Ошибка при загрузке (попытка {attempt + 1}): {e}")
        print("Все попытки загрузки завершились неудачно")
        return None

    @timer_decorator
    def download_images(self) -> None:
        """Загрузить изображения из API."""
        try:
            api_data = self._fetch_images_from_api()
            
            successful_downloads = 0
            for i, image_data in enumerate(api_data):
                print(f"\n{'='*50}")
                print(f" ЗАГРУЗКА ИЗОБРАЖЕНИЯ {i+1}/{len(api_data)}")
                print(f"{'='*50}")
                
                image_url = image_data['url']
                breed = self._get_breed_name(image_data)
                
                print(f"Порода: {breed}")
                print(f"URL: {image_url}")
                
                image_array = self._download_image(image_url)
                if image_array is not None:
                    cat_image = ColorCatImage(image_array, image_url, breed)
                    image_type = "цветное"
                    
                    self._downloaded_images.append(cat_image)
                    successful_downloads += 1
                    print(f"УСПЕШНО ЗАГРУЖЕНО: {breed} ({image_type})")
                    print(f"Строковое представление: {cat_image}")
                else:
                    print(f"НЕ УДАЛОСЬ ЗАГРУЗИТЬ изображение {i+1}")
            
            print(f"\n{'='*50}")
            print(f"ИТОГ ЗАГРУЗКИ")
            print(f"{'='*50}")
            print(f"O Успешно: {successful_downloads}/{len(api_data)}")
            print(f"X Неудачно: {len(api_data) - successful_downloads}/{len(api_data)}")
            
        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА при загрузке изображений: {e}")
            raise

    @timer_decorator
    def _get_breed_name(self, image_data: dict) -> str:
        """
        Извлечь название породы из данных изображения.
        """
        breeds = image_data.get('breeds', [])
        if breeds and len(breeds) > 0:
            breed_name = breeds[0].get('name', 'unknown')
            breed_name = breed_name.replace(' ', '_').replace('/', '_').lower()
            return breed_name
        return 'unknown'
    
    @timer_decorator
    def _save_image(self, image: np.ndarray, filename: str) -> None:
        """
        Сохранить изображение в файл.
        """
        try:
            filepath = os.path.join(self._output_dir, filename)
            success = cv2.imwrite(filepath, image)
            if success:
                print(f"Изображение сохранено: {filepath}")
            else:
                print(f"X Ошибка при сохранении: {filepath}")
        except Exception as e:
            print(f"X Ошибка при сохранении изображения {filename}: {e}")
    
    @timer_decorator
    def process_images(self) -> None:
        """Обработать все загруженные изображения."""
        if not self._downloaded_images:
            print("Нет загруженных изображений. Загружаем...")
            self.download_images()
        
        if not self._downloaded_images:
            print("X Нет изображений для обработки")
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
                base_filename = f"{i+1}_{cat_image.breed}"
                
                # Исходное изображение
                original_filename = f"{base_filename}_original.png"
                self._save_image(cat_image.image_data, original_filename)
                
                # Контуры пользовательским методом
                custom_filename = f"{base_filename}_custom_edges.png"
                self._save_image(custom_edges, custom_filename)
                
                # Контуры библиотечным методом
                library_filename = f"{base_filename}_library_edges.png"
                self._save_image(library_edges, library_filename)
                
                print(f"O Завершена обработка изображения {i+1}")
                
            except Exception as e:
                print(f"X Ошибка при обработке изображения {i+1}: {e}")
                continue
    
    @timer_decorator
    def demonstrate_operations(self) -> None:
        """
        Продемонстрировать операции сложения и вычитания между всеми изображениями.
        """
        if len(self._downloaded_images) < 2:
            print("Для демонстрации операций нужно как минимум 2 изображения")
            return
            
        print("\n--- Демонстрация операций с изображениями ---")
        
        try:
            # Демонстрируем операции между разными парами изображений
            operation_count = 0
            
            for i in range(len(self._downloaded_images)):
                for j in range(i + 1, len(self._downloaded_images)):
                    if operation_count >= 3:  # Ограничим количество операций
                        break
                        
                    img1 = self._downloaded_images[i]
                    img2 = self._downloaded_images[j]
                    
                    print(f"\nОперация {operation_count + 1}: {img1.breed} и {img2.breed}")
                    
                    # Сложение
                    print("Выполняем сложение...")
                    added = img1 + img2
                    print(f"Результат сложения: {added}")
                    
                    # Вычитание
                    print("Выполняем вычитание...")
                    subtracted = img1 - img2
                    print(f"Результат вычитания: {subtracted}")
                    
                    # Сохранение результатов операций
                    self._save_image(added.image_data, f"addition_{operation_count + 1}_{img1.breed}_{img2.breed}.png")
                    self._save_image(subtracted.image_data, f"subtraction_{operation_count + 1}_{img1.breed}_{img2.breed}.png")
                    
                    operation_count += 1
                    if operation_count >= 3:
                        break
            
            print(f"\nO Выполнено {operation_count} операций между разными изображениями")
            print("Демонстрационные изображения сохранены")
            
        except Exception as e:
            print(f"X Ошибка при демонстрации операций: {e}")