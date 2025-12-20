"""
async_cat_image_processor_v2.py

Улучшенная асинхронная версия с асинхронным логированием для лабораторной работы №5.
"""

import os
import asyncio
import aiohttp
import aiofiles
import numpy as np
from typing import List, Optional, AsyncGenerator, Tuple, Dict
import time
import cv2
from concurrent.futures import ProcessPoolExecutor

# Импортируем конфигурацию и классы изображений
from .config import API_KEY, BASE_URL, DEFAULT_LIMIT, DEFAULT_OUTPUT_DIR
from .cat_image import CatImage, create_cat_image
from .async_logging_config import get_async_logger


class AsyncCatImageProcessorV2:
    """
    Улучшенный асинхронный класс с асинхронным логированием.
    """
    
    def __init__(self, limit: int = DEFAULT_LIMIT, output_dir: str = DEFAULT_OUTPUT_DIR):
        """
        Инициализация асинхронного процессора.
        """
        self._limit = limit
        self._api_key = API_KEY
        self._base_url = BASE_URL
        self._downloaded_images: List[CatImage] = []
        self._output_dir = output_dir
        self._max_image_size = 1500
        self._start_time = 0
        self._images_dict: Dict[int, CatImage] = {}  # Словарь для хранения изображений по индексу
        
        # Получаем логгер
        self._logger = get_async_logger("async_processor_v2")
        
    @property
    def downloaded_images(self) -> List[CatImage]:
        """Получить список загруженных изображений."""
        return self._downloaded_images

    async def _create_output_directory(self) -> None:
        """Создать директорию для сохранения изображений."""
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)
            self._logger.debug(f"Создана директория: {self._output_dir}")

    async def fetch_image_urls(self) -> AsyncGenerator[Tuple[int, str, str], None]:
        """
        Асинхронный генератор для получения URL изображений.
        """
        self._logger.info(f"Получение {self._limit} URL изображений из API...")
        
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
                                    
                                    self._logger.debug(f"URL для изображения {fetched_count} получен: {breed}")
                        else:
                            self._logger.warning(f"API недоступен, используем демо-режим")
                            break
                            
                except Exception as e:
                    self._logger.error(f"Ошибка при запросе к API: {e}")
                    break
            
            if fetched_count < self._limit:
                demo_urls = await self._get_demo_urls(self._limit - fetched_count)
                for url, breed in demo_urls:
                    if fetched_count < self._limit:
                        yield fetched_count, url, breed
                        fetched_count += 1
                        self._logger.debug(f"Демо URL для изображения {fetched_count} получен: {breed}")

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
        self._logger.info("  Этап скачивания запущен")
        
        download_stats = {"total": 0, "success": 0, "failed": 0}
        
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False)) as session:
            async for idx, url, breed in url_generator:
                download_stats["total"] += 1
                start_time = time.time()
                self._logger.debug(f"⬇️ Скачивание {idx+1} начато: {breed}")
                
                result = await self._download_with_retry(session, idx, url, breed, max_retries=10)
                if result is not None:
                    image_data, final_url, final_breed = result
                    height, width = image_data.shape[:2]
                    end_time = time.time()
                    download_time = end_time - start_time
                    download_stats["success"] += 1
                    
                    # Считаем попытки (если порода изменилась, значит были попытки)
                    attempts_info = ""
                    if final_breed != breed:
                        attempts_info = " (смена породы)"
                    
                    # INFO: только результат
                    self._logger.info(f"     ✅ Скачивание {idx+1} завершено: {final_breed} - {width}x{height} "
                                    f"за {download_time:.2f} сек{attempts_info}")
                    
                    # DEBUG: детали если порода изменилась
                    if final_breed != breed:
                        self._logger.debug(f"Порода изображения {idx+1} изменена: '{breed}' → '{final_breed}'")
                    
                    yield idx, image_data, final_url, final_breed
                else:
                    download_stats["failed"] += 1
                    self._logger.error(f"❌ Скачивание {idx+1} неудачно после всех попыток")
        
        self._logger.info(f"📊 Скачивание завершено: {download_stats['success']}/{download_stats['total']} "
                        f"успешно, {download_stats['failed']} неудачно")

    async def _download_with_retry(self, session: aiohttp.ClientSession, idx: int, 
                              url: str, breed: str, max_retries: int = 10) -> Optional[Tuple[np.ndarray, str, str]]:
        """
        Скачать изображение с повторными попытками при слишком больших размерах.
        Возвращает кортеж (image_data, url, breed) или None.
        """
        original_breed = breed
        attempts_made = 0
        
        for attempt in range(max_retries):
            attempts_made += 1
            current_url = url
            current_breed = breed
            
            try:
                image_data = await self._download_single_image(session, idx, current_url, current_breed, attempt)
                if image_data is not None:
                    height, width = image_data.shape[:2]
                    
                    if height <= self._max_image_size and width <= self._max_image_size:
                        
                        
                        # INFO: только если порода изменилась
                        if current_breed != original_breed:
                            self._logger.debug(f"✅ Изображение {idx+1}: порода изменена '{original_breed}' → '{current_breed}'")
                        else:
                            self._logger.debug(f"Изображение {idx+1} подходящего размера: {width}x{height}, порода: {current_breed}")

                        return image_data, current_url, current_breed
                    else:
                        # DEBUG: детали о большом размере
                        self._logger.debug(f"Изображение {idx+1} слишком большое: {width}x{height} > {self._max_image_size} "
                                        f"(попытка {attempt + 1}/{max_retries}, порода: {current_breed})")
                        
                        if attempt < max_retries - 1:
                            # Ждем перед повторной попыткой, чтобы избежать rate limiting
                            await asyncio.sleep(1)
                            new_url_data = await self._fetch_single_image_url(session)
                            if new_url_data:
                                old_breed = current_breed
                                url = new_url_data['url']
                                breed = self._get_breed_name(new_url_data)
                                
                                # DEBUG: детали о смене URL
                                self._logger.debug(f"🔄 Смена URL для изображения {idx+1}: "
                                                f"'{old_breed}' → '{breed}'")
                            else:
                                self._logger.debug(f"Не удалось получить новый URL для изображения {idx+1}")
                                break
                else:
                    self._logger.debug(f"Не удалось загрузить изображение {idx+1} (попытка {attempt + 1}/{max_retries})")
                    
            except Exception as e:
                self._logger.error(f"Ошибка при загрузке изображения {idx+1} (попытка {attempt + 1}/{max_retries}): {e}")
        
        self._logger.error(f"Все {max_retries} попытки загрузки изображения {idx+1} завершились неудачно")
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
                        self._logger.warning(f"Пустой контент для изображения {idx+1}")
                        return None
                    
                    image_array = np.frombuffer(content, np.uint8)
                    image_array = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    
                    if image_array is not None:
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                        return image_array
                            
        except Exception as e:
            self._logger.error(f"Ошибка при загрузке изображения {idx+1} (попытка {attempt + 1}): {e}")
        
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
            # Ждем перед запросом, чтобы избежать rate limiting
            await asyncio.sleep(1)
            
            async with session.get(
                f"{self._base_url}/images/search",
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        breed_name = self._get_breed_name(data[0])
                        self._logger.debug(f"Получен новый URL изображения с породой: {breed_name}")
                        return data[0]
                    else:
                        self._logger.warning("API вернул пустой ответ при запросе нового URL")
                elif response.status == 429:
                    self._logger.warning(f"Rate limit достигнут, ждем 5 секунд...")
                    await asyncio.sleep(5)
                else:
                    self._logger.warning(f"API вернул статус {response.status} при запросе нового URL")
        
        except Exception as e:
            self._logger.warning(f"Ошибка при получении нового URL: {e}")
        
        return None

    async def process_images(self, download_generator: AsyncGenerator) -> AsyncGenerator[Tuple[int, CatImage], None]:
        """
        Асинхронный генератор для обработки изображений в параллельных процессах.
        """
        self._logger.info("  Этап обработки запущен")
        
        loop = asyncio.get_event_loop()
        max_workers = min(4, os.cpu_count() or 2)
        
        self._logger.debug(f"Используется {max_workers} воркеров для обработки")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Создаем задачи обработки по мере поступления скачанных изображений
            processing_tasks = []
            task_start_times = {}
            processing_stats = {"total": 0, "success": 0, "failed": 0}
            
            async for idx, image_data, url, breed in download_generator:
                processing_stats["total"] += 1
                start_time = time.time()
                task_start_times[idx] = start_time
                
                # INFO: начало обработки
                self._logger.debug(f"⚙️ Обработка {idx+1} начата: {breed} ({image_data.shape[1]}x{image_data.shape[0]})")
                
                # НЕМЕДЛЕННО запускаем обработку в отдельном процессе
                task = asyncio.create_task(
                    self._process_single_image_async(loop, executor, idx, image_data, url, breed)
                )
                processing_tasks.append(task)
                
                # Ограничиваем количество одновременно обрабатываемых изображений
                if len(processing_tasks) >= max_workers:
                    # Ждем завершения одной задачи перед добавлением новой
                    try:
                        done, pending = await asyncio.wait(processing_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=30)
                        
                        for task in done:
                            try:
                                result = await task
                                if result is not None:
                                    idx_result, cat_image = result
                                    processing_stats["success"] += 1
                                    
                                    # Сохраняем изображение в словарь
                                    self._images_dict[idx_result] = cat_image
                                    
                                    # Вычисляем время обработки
                                    end_time = time.time()
                                    processing_time = end_time - task_start_times.get(idx_result, end_time)
                                    
                                    # INFO: результат обработки
                                    self._logger.info(f"    ⚙️ Обработка {idx_result+1} завершена: {cat_image.breed} - "
                                                    f"за {processing_time:.2f} сек")
                                    
                                    yield idx_result, cat_image
                                else:
                                    processing_stats["failed"] += 1
                                    self._logger.error(f"❌ Ошибка при обработке изображения {idx+1}")
                            except Exception as e:
                                processing_stats["failed"] += 1
                                self._logger.error(f"❌ Исключение при обработке изображения: {e}")
                        
                        # Удаляем завершенные задачи
                        processing_tasks = list(pending)
                        
                    except asyncio.TimeoutError:
                        self._logger.warning("Таймаут ожидания завершения обработки")
            
            # Обрабатываем оставшиеся задачи
            if processing_tasks:
                self._logger.debug(f"Обработка оставшихся {len(processing_tasks)} задач...")
                for completed_task in asyncio.as_completed(processing_tasks):
                    try:
                        result = await completed_task
                        if result is not None:
                            idx_result, cat_image = result
                            processing_stats["success"] += 1
                            
                            # Сохраняем изображение в словарь
                            self._images_dict[idx_result] = cat_image
                            
                            # Вычисляем время обработки
                            end_time = time.time()
                            processing_time = end_time - task_start_times.get(idx_result, end_time)
                            
                            # INFO: результат обработки
                            self._logger.info(f"    ⚙️ Обработка {idx_result+1} завершена: {cat_image.breed} - "
                                            f"за {processing_time:.2f} сек")
                            
                            yield idx_result, cat_image
                        else:
                            processing_stats["failed"] += 1
                            self._logger.error(f"❌ Ошибка при обработке изображения")
                    except Exception as e:
                        processing_stats["failed"] += 1
                        self._logger.error(f"❌ Исключение при обработке изображения: {e}")
            
            self._logger.info(f"📊 Обработка завершена: {processing_stats['success']}/{processing_stats['total']} "
                            f"успешно, {processing_stats['failed']} неудачно")

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
            self._logger.error(f"Ошибка при асинхронной обработке изображения {idx+1}: {e}")
            return None

    def _process_single_image(self, idx: int, image_data: np.ndarray, url: str, breed: str) -> Optional[Tuple[int, CatImage]]:
        """
        Обработка одного изображения в параллельном процессе.
        """
        current_pid = os.getpid()
        start_time = time.time()
        
        # DEBUG: детали начала в процессе
        self._logger.debug(f"Convolution for image {idx+1} started (PID {current_pid}) - {image_data.shape[1]}x{image_data.shape[0]}")
        
        try:
            cat_image = create_cat_image(image_data, url, breed)
            custom_edges, library_edges = cat_image.process_edges()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # DEBUG: детали завершения в процессе
            self._logger.debug(f"Convolution for image {idx+1} finished (PID {current_pid}) - "
                            f"за {processing_time:.2f} сек")
            return idx, cat_image
            
        except Exception as e:
            self._logger.error(f"Ошибка при обработке изображения {idx+1} в процессе {current_pid}: {e}")
            return None

    async def save_images(self, process_generator: AsyncGenerator) -> AsyncGenerator[Tuple[int, CatImage], None]:
        """
        Асинхронный генератор для сохранения изображений.
        Сохранение запускается НЕМЕДЛЕННО после обработки каждого изображения.
        Возвращает (индекс, изображение) сразу после сохранения.
        """
        self._logger.info("  Этап сохранения запущен")
        await self._create_output_directory()
        
        save_stats = {"total": 0, "success": 0, "failed": 0}
        save_tasks = []
        pipeline_start_time = self._start_time
        
        async for idx, cat_image in process_generator:
            save_stats["total"] += 1
            start_time = time.time()
            
            # Сохраняем изображение в список
            self._downloaded_images.append(cat_image)
            
            # INFO: начало сохранения
            self._logger.debug(f"💾 Сохранение {idx+1} начато: {cat_image.breed}")
            
            # НЕМЕДЛЕННО запускаем сохранение
            task = asyncio.create_task(self._save_single_image_batch_with_time(idx, cat_image, start_time, pipeline_start_time))
            save_tasks.append(task)
            
            # Ограничиваем количество одновременно сохраняемых изображений
            if len(save_tasks) >= 3:  # Максимум 3 одновременных сохранения
                # Ждем завершения одной задачи
                try:
                    done, pending = await asyncio.wait(save_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=30)
                    
                    for task_obj in done:
                        try:
                            result = await task_obj
                            if result is not None:
                                idx_result, saved_successfully = result
                                if saved_successfully:
                                    save_stats["success"] += 1
                                else:
                                    save_stats["failed"] += 1
                                    
                                # Возвращаем изображение сразу после сохранения
                                yield idx_result, self._images_dict.get(idx_result)
                            else:
                                save_stats["failed"] += 1
                        except Exception as e:
                            save_stats["failed"] += 1
                            self._logger.error(f"Ошибка при сохранении: {e}")
                    
                    # Удаляем завершенные задачи
                    save_tasks = list(pending)
                    
                except asyncio.TimeoutError:
                    self._logger.warning("Таймаут ожидания завершения сохранения")
        
        # Обрабатываем оставшиеся задачи сохранения
        if save_tasks:
            for completed_task in asyncio.as_completed(save_tasks):
                try:
                    result = await completed_task
                    if result is not None:
                        idx_result, saved_successfully = result
                        if saved_successfully:
                            save_stats["success"] += 1
                        else:
                            save_stats["failed"] += 1
                            
                        # Возвращаем изображение сразу после сохранения
                        yield idx_result, self._images_dict.get(idx_result)
                    else:
                        save_stats["failed"] += 1
                except Exception as e:
                    save_stats["failed"] += 1
                    self._logger.error(f"Исключение при сохранении изображения: {e}")
        
        self._logger.info(f"📊 Сохранение завершено: {save_stats['success']}/{save_stats['total']} "
                        f"успешно, {save_stats['failed']} неудачно")

    async def _save_single_image_batch_with_time(self, idx: int, cat_image: CatImage, 
                                               start_time: float, pipeline_start_time: float) -> Tuple[int, bool]:
        """Сохранить все файлы для одного изображения с измерением времени и выводом финального сообщения."""
        try:
            await self._save_single_image_batch(idx, cat_image)
            end_time = time.time()
            save_time = end_time - start_time
            total_time = end_time - pipeline_start_time
            
            # INFO: завершение сохранения
            self._logger.info(f"   💾 Сохранение {idx+1} завершено - за {save_time:.2f} сек")
            
            # Сразу выводим финальное сообщение
            self._logger.info(f"  🎯 Изображение {idx+1} полностью готово: {cat_image.breed} "
                            f"({cat_image.image_data.shape[1]}x{cat_image.image_data.shape[0]}) - "
                            f"прошло {total_time:.2f} сек")
            
            return idx, True
        except Exception as e:
            self._logger.error(f"❌ Ошибка при сохранении изображения {idx+1}: {e}")
            return idx, False

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
            self._logger.warning(f"Ошибка при операциях с изображением {idx+1}: {e}")
        
        results = await asyncio.gather(*save_tasks, return_exceptions=True)
        
        # Логируем результаты сохранения
        success_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self._logger.warning(f"Ошибка при сохранении файла {i+1} для изображения {idx+1}: {result}")
            elif result:
                success_count += 1
        
        self._logger.debug(f"Сохранено {success_count}/{len(save_tasks)} файлов для изображения {idx+1}")

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
                self._logger.debug(f"Файл сохранен: {filename}")
                return True
                
        except Exception as e:
            self._logger.warning(f"Ошибка при сохранении {filename}: {e}")
        
        return False

    async def run_pipeline(self) -> None:
        """
        Запуск полного асинхронного генераторного пайплайна.
        Все этапы работают ПАРАЛЛЕЛЬНО.
        """
        self._logger.info("🚀 Запуск асинхронного генераторного пайплайна...")
        self._logger.info("💡 Этапы работают параллельно: скачивание ↔ обработка ↔ сохранение")
        self._start_time = time.time()
        pipeline_start_time = self._start_time
        
        try:
            # Создаем генераторный пайплайн
            url_generator = self.fetch_image_urls()
            download_generator = self.download_images(url_generator)
            process_generator = self.process_images(download_generator)
            save_generator = self.save_images(process_generator)
            
            # Запускаем пайплайн
            processed_count = 0
            async for idx, cat_image in save_generator:
                if cat_image is not None:
                    processed_count += 1
                    # Сообщение "Изображение полностью готово" теперь выводится сразу в методе сохранения
                    # Здесь просто увеличиваем счетчик
            
            total_time = time.time() - pipeline_start_time
            
            # Вывод итоговой статистики
            self._logger.info("=" * 60)
            self._logger.info("📊 ИТОГОВАЯ СТАТИСТИКА:")
            self._logger.info("=" * 60)
            self._logger.info(f"🎯 Всего изображений: {processed_count}")
            self._logger.info(f"⏱️ Общее время: {total_time:.2f} секунд")
            self._logger.info(f"📈 Среднее время на изображение: {total_time/max(1, processed_count):.2f} секунд")
            self._logger.info("=" * 60)
            
        except Exception as e:
            self._logger.error(f"💥 Критическая ошибка в пайплайне: {e}")
            import traceback
            self._logger.error(f"Трассировка ошибки:\n{traceback.format_exc()}")
            elapsed_time = time.time() - pipeline_start_time
            self._logger.warning(f"⏱️ Прошло времени: {elapsed_time:.2f} секунд")
            
            # Вывод статистики даже при ошибке
            processed_count = len(self._downloaded_images)
            if processed_count > 0:
                self._logger.info(f"📊 Удалось обработать {processed_count} изображений из {self._limit}")


if __name__ == "__main__":
    # При запуске напрямую
    async def main():
        processor = AsyncCatImageProcessorV2(limit=2)
        await processor.run_pipeline()
    
    asyncio.run(main())