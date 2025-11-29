"""
Асинхронный модуль для работы с API и обработки изображений животных.
"""
import os
import asyncio
import aiohttp
import aiofiles
import numpy as np
from typing import List, Optional, AsyncGenerator, Tuple
import time
import cv2
from concurrent.futures import ProcessPoolExecutor

# Импортируем конфигурацию и классы изображений
from config import API_KEY, BASE_URL, DEFAULT_LIMIT, DEFAULT_OUTPUT_DIR
from cat_image import CatImage, create_cat_image


class AsyncCatImageProcessor:
    """
    Асинхронный класс для обработки изображений животных через API.
    Реализует асинхронный генераторный пайплайн.
    """
    
    def __init__(self, limit: int = DEFAULT_LIMIT, output_dir: str = DEFAULT_OUTPUT_DIR):
        """
        Инициализация асинхронного процессора.
        
        Args:
            limit: Количество изображений для загрузки
            output_dir: Директория для сохранения результатов
        """
        self._limit = limit
        self._api_key = API_KEY
        self._base_url = BASE_URL
        self._downloaded_images: List[CatImage] = []
        self._output_dir = output_dir
        self._max_image_size = 1500
        self._start_time = 0
        
    @property
    def downloaded_images(self) -> List[CatImage]:
        """Получить список загруженных изображений."""
        return self._downloaded_images

    async def _create_output_directory(self) -> None:
        """Создать директорию для сохранения изображений."""
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)
            print(f"Создана директория: {self._output_dir}")

    async def fetch_image_urls(self) -> AsyncGenerator[Tuple[int, str, str], None]:
        """
        Асинхронный генератор для получения URL изображений.
        """
        print(f"🔄 Получение {self._limit} URL изображений из API...")
        
        batch_size = 10
        remaining = self._limit
        fetched_count = 0
        
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False)) as session:
            while remaining > 0:
                current_batch = min(batch_size, remaining)
                
                params = {
                    'limit': current_batch, 
                    'has_breeds': 1,
                    'size': 'med'
                }
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                    'x-api-key': self._api_key
                }
                
                try:
                    async with session.get(
                        f"{self._base_url}/images/search",
                        params=params,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        
                        if response.status == 200:
                            data = await response.json()
                            for image_data in data:
                                if fetched_count < self._limit:
                                    image_url = image_data['url']
                                    breed = self._get_breed_name(image_data)
                                    
                                    yield fetched_count, image_url, breed
                                    fetched_count += 1
                                    remaining -= 1
                                    
                                    print(f"📋 URL для изображения {fetched_count} получен: {breed}")
                        else:
                            print(f"⚠️ API недоступен, используем демо-режим")
                            break
                            
                except Exception as e:
                    print(f"⚠️ Ошибка при запросе к API: {e}")
                    break
            
            if fetched_count < self._limit:
                demo_urls = await self._get_demo_urls(self._limit - fetched_count)
                for url, breed in demo_urls:
                    if fetched_count < self._limit:
                        yield fetched_count, url, breed
                        fetched_count += 1
                        print(f"📋 Демо URL для изображения {fetched_count} получен: {breed}")

    async def _get_demo_urls(self, count: int) -> List[Tuple[str, str]]:
        """Генерирует демо-URL для тестирования."""
        base_demo_data = [
            ('https://cdn2.thecatapi.com/images/9mNopQrStU.jpg', 'russian_blue'),
            ('https://cdn2.thecatapi.com/images/9u1.jpg', 'abyssinian'),
            ('https://cdn2.thecatapi.com/images/bt.jpg', 'bengal'),
        ]
        
        demo_urls = []
        for i in range(count):
            url, breed = base_demo_data[i % len(base_demo_data)]
            demo_urls.append((f"{url}?demo={i}", f"{breed}_{i}"))
        
        return demo_urls

    def _get_breed_name(self, image_data: dict) -> str:
        """Извлечь название породы из данных изображения."""
        breeds = image_data.get('breeds', [])
        if breeds and len(breeds) > 0:
            breed_name = breeds[0].get('name', 'unknown')
            breed_name = breed_name.replace(' ', '_').replace('/', '_').replace('\\', '_').lower()
            return breed_name
        return 'unknown'

    async def download_images(self, url_generator: AsyncGenerator) -> AsyncGenerator[Tuple[int, np.ndarray, str, str], None]:
        """
        Асинхронный генератор для скачивания изображений с повторными попытками.
        """
        print("🎯 Этап скачивания запущен")
        
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False)) as session:
            async for idx, url, breed in url_generator:
                print(f"📥 Downloading image {idx+1} started")
                
                image_data = await self._download_with_retry(session, idx, url, breed, max_retries=10)
                if image_data is not None:
                    height, width = image_data.shape[:2]
                    print(f"✅ Downloading image {idx+1} finished - {width}x{height}")
                    yield idx, image_data, url, breed
                else:
                    print(f"❌ Downloading image {idx+1} failed после всех попыток")

    async def _download_with_retry(self, session: aiohttp.ClientSession, idx: int, 
                              url: str, breed: str, max_retries: int = 3) -> Optional[np.ndarray]:
        """
        Скачать изображение с повторными попытками при слишком больших размерах.
        """
        for attempt in range(max_retries):
            try:
                image_data = await self._download_single_image(session, idx, url, breed, attempt)
                if image_data is not None:
                    height, width = image_data.shape[:2]
                    
                    if height <= self._max_image_size and width <= self._max_image_size:
                        print(f"✅ Изображение {idx+1} подходящего размера: {width}x{height}")
                        return image_data
                    else:
                        print(f"⚠️ Изображение {idx+1} слишком большое: {width}x{height} > {self._max_image_size} (попытка {attempt + 1}/{max_retries})")
                        
                        if attempt < max_retries - 1:
                            new_url_data = await self._fetch_single_image_url(session)
                            if new_url_data:
                                url = new_url_data['url']
                                breed = self._get_breed_name(new_url_data)  # ✅ Извлекаем породу из новых данных
                                print(f"🔄 Новый URL для изображения {idx+1}: {breed}")
                            else:
                                print(f"❌ Не удалось получить новый URL для изображения {idx+1}")
                                break
                else:
                    print(f"❌ Не удалось загрузить изображение {idx+1} (попытка {attempt + 1}/{max_retries})")
                    
            except Exception as e:
                print(f"❌ Ошибка при загрузке изображения {idx+1} (попытка {attempt + 1}/{max_retries}): {e}")
        
        print(f"❌ Все {max_retries} попытки загрузки изображения {idx+1} завершились неудачно")
        return None

    async def _download_single_image(self, session: aiohttp.ClientSession, idx: int, 
                                   url: str, breed: str, attempt: int = 0) -> Optional[np.ndarray]:
        """Скачать одно изображение асинхронно."""
        timeout = aiohttp.ClientTimeout(total=30)
        
        try:
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    if not content:
                        return None
                    
                    image_array = np.frombuffer(content, np.uint8)
                    image_array = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    
                    if image_array is not None:
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                        return image_array
                            
        except Exception as e:
            print(f"❌ Ошибка при загрузке изображения {idx+1} (попытка {attempt + 1}): {e}")
        
        return None

    async def _fetch_single_image_url(self, session: aiohttp.ClientSession) -> Optional[dict]:
        """Получить URL одного изображения из API."""
        params = {
            'limit': 1, 
            'has_breeds': 1,
            'size': 'med'
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'x-api-key': self._api_key
        }
        
        try:
            async with session.get(
                f"{self._base_url}/images/search",
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        return data[0]
        except Exception as e:
            print(f"⚠️ Ошибка при получении нового URL: {e}")
        
        return None

    async def process_images(self, download_generator: AsyncGenerator) -> AsyncGenerator[Tuple[int, CatImage], None]:
        """
        Асинхронный генератор для обработки изображений в параллельных процессах.
        Обработка запускается НЕМЕДЛЕННО после скачивания каждого изображения.
        """
        print("🔄 Этап обработки запущен")
        
        loop = asyncio.get_event_loop()
        max_workers = min(8, os.cpu_count() or 4)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Создаем задачи обработки по мере поступления скачанных изображений
            processing_tasks = []
            
            async for idx, image_data, url, breed in download_generator:
                # НЕМЕДЛЕННО запускаем обработку в отдельном процессе
                task = asyncio.create_task(
                    self._process_single_image_async(loop, executor, idx, image_data, url, breed)
                )
                processing_tasks.append(task)
                print(f"⚡ Задача обработки для изображения {idx+1} запущена")
            
            # Обрабатываем результаты по мере готовности
            for completed_task in asyncio.as_completed(processing_tasks):
                try:
                    result = await completed_task
                    if result is not None:
                        idx, cat_image = result
                        yield idx, cat_image
                except Exception as e:
                    print(f"❌ Ошибка при обработке изображения: {e}")

    async def _process_single_image_async(self, loop, executor, idx: int, image_data: np.ndarray, url: str, breed: str) -> Optional[Tuple[int, CatImage]]:
        """
        Асинхронная обертка для обработки одного изображения.
        """
        try:
            result = await loop.run_in_executor(
                executor, self._process_single_image, idx, image_data, url, breed
            )
            return result
        except Exception as e:
            print(f"❌ Ошибка при асинхронной обработке изображения {idx+1}: {e}")
            return None

    def _process_single_image(self, idx: int, image_data: np.ndarray, url: str, breed: str) -> Optional[Tuple[int, CatImage]]:
        """
        Обработка одного изображения в параллельном процессе.
        """
        current_pid = os.getpid()
        print(f"⚡ Convolution for image {idx+1} started (PID {current_pid})")
        
        try:
            cat_image = create_cat_image(image_data, url, breed)
            custom_edges, library_edges = cat_image.process_edges()
            
            print(f"✅ Convolution for image {idx+1} finished (PID {current_pid})")
            return idx, cat_image
            
        except Exception as e:
            print(f"❌ Ошибка при обработке изображения {idx+1} в процессе {current_pid}: {e}")
            return None

    async def save_images(self, process_generator: AsyncGenerator) -> AsyncGenerator[Tuple[int, CatImage], None]:
        """
        Асинхронный генератор для сохранения изображений.
        Сохранение запускается НЕМЕДЛЕННО после обработки каждого изображения.
        """
        print("💾 Этап сохранения запущен")
        await self._create_output_directory()
        
        async for idx, cat_image in process_generator:
            print(f"💾 Saving image {idx+1} started")
            
            try:
                await self._save_single_image_batch(idx, cat_image)
                print(f"✅ Saving image {idx+1} finished")
                
                self._downloaded_images.append(cat_image)
                yield idx, cat_image
                
            except Exception as e:
                print(f"❌ Ошибка при сохранении изображения {idx+1}: {e}")

    async def _save_single_image_batch(self, idx: int, cat_image: CatImage) -> None:
        """Сохранить все файлы для одного изображения."""
        base_filename = f"{idx+1:02d}_{cat_image.breed}"
        save_tasks = []
        
        original_filename = f"{base_filename}_original.png"
        save_tasks.append(self._save_single_image(cat_image.image_data, original_filename))
        
        if cat_image.processed_edges_custom is not None:
            custom_filename = f"{base_filename}_custom_edges.png"
            save_tasks.append(self._save_single_image(cat_image.processed_edges_custom, custom_filename))
        
        if cat_image.processed_edges_library is not None:
            library_filename = f"{base_filename}_library_edges.png"
            save_tasks.append(self._save_single_image(cat_image.processed_edges_library, library_filename))
        
        try:
            added_custom_result = cat_image + 'custom'
            addition_filename = f"{base_filename}_ORIGINAL_PLUS_CUSTOM.png"
            save_tasks.append(self._save_single_image(added_custom_result.image_data, addition_filename))
            
            subtracted_custom_result = cat_image - 'custom'
            subtraction_filename = f"{base_filename}_ORIGINAL_MINUS_CUSTOM.png"
            save_tasks.append(self._save_single_image(subtracted_custom_result.image_data, subtraction_filename))
        except Exception as e:
            print(f"⚠️ Ошибка при операциях с изображением {idx+1}: {e}")
        
        await asyncio.gather(*save_tasks, return_exceptions=True)

    async def _save_single_image(self, image: np.ndarray, filename: str) -> bool:
        """Асинхронно сохранить одно изображение."""
        try:
            filepath = os.path.join(self._output_dir, filename)
            
            if len(image.shape) == 2:
                success, encoded_image = cv2.imencode('.png', image)
            else:
                if image.shape[2] == 3:
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    success, encoded_image = cv2.imencode('.png', image_bgr)
                else:
                    success, encoded_image = cv2.imencode('.png', image)
            
            if success:
                async with aiofiles.open(filepath, 'wb') as f:
                    await f.write(encoded_image.tobytes())
                return True
                
        except Exception as e:
            print(f"⚠️ Ошибка при сохранении {filename}: {e}")
        
        return False

    async def run_pipeline(self) -> None:
        """
        Запуск полного асинхронного генераторного пайплайна.
        Все этапы работают ПАРАЛЛЕЛЬНО.
        """
        print("🚀 Запуск асинхронного генераторного пайплайна...")
        print("💡 Этапы работают синхронно: скачивание ↔ обработка (параллельна внутри) ↔ сохранение")
        self._start_time = time.time()
        
        try:
            # Создаем генераторный пайплайн
            url_generator = self.fetch_image_urls()
            download_generator = self.download_images(url_generator)
            process_generator = self.process_images(download_generator)
            save_generator = self.save_images(process_generator)
            
            # Запускаем пайплайн - все этапы работают параллельно
            processed_count = 0
            async for idx, cat_image in save_generator:
                processed_count += 1
                print(f"🎯 Изображение {idx+1} полностью обработано и сохранено")
                
            elapsed_time = time.time() - self._start_time
            print(f"🎉 Пайплайн завершен за {elapsed_time:.2f} секунд")
            print(f"📊 Обработано {processed_count} изображений")
            
        except Exception as e:
            print(f"💥 Критическая ошибка в пайплайне: {e}")
            elapsed_time = time.time() - self._start_time
            print(f"⏱️ Прошло времени: {elapsed_time:.2f} секунд")


async def benchmark_performance():
    """Сравнение производительности."""
    print("🧪 Запуск теста производительности...")
    
    for limit in [2, 3]:
        print(f"\n{'='*50}")
        print(f"Тест с {limit} изображениями")
        print(f"{'='*50}")
        
        async_processor = AsyncCatImageProcessor(limit=limit)
        start_time = time.time()
        await async_processor.run_pipeline()
        async_time = time.time() - start_time
        
        print(f"⏱️ Асинхронная версия: {async_time:.2f} секунд")
        print(f"📊 Обработано: {len(async_processor.downloaded_images)} изображений")


if __name__ == "__main__":
    asyncio.run(benchmark_performance())