"""
Модуль image_processing.py

Реализация интерфейса IImageProcessing с использованием библиотеки OpenCV.

Содержит класс ImageProcessing, предоставляющий методы для обработки изображений:
- свёртка изображения с ядром
- преобразование RGB-изображения в оттенки серого
- гамма-коррекция
- обнаружение границ (оператор Кэнни)
- обнаружение углов (алгоритм Харриса)
- обнаружение окружностей (метод пока не реализован)

Модуль предназначен для учебных целей (лабораторная работа по курсу "Технологии программирования на Python").
"""
import time
import cv2
import interfaces
import numpy as np


class ImageProcessing(interfaces.IImageProcessing):
    """
    Реализация интерфейса IImageProcessing с использованием библиотеки OpenCV.

    Предоставляет методы для обработки изображений, включая свёртку, преобразование
    в оттенки серого, гамма-коррекцию, а также обнаружение границ, углов и окружностей.
    """

    def convolution(self, image: np.ndarray, kernel: np.ndarray) -> tuple:
        """
        Выполняет свёртку изображения с заданным ядром.

        Args:
            image (np.ndarray): Входное изображение (может быть цветным или чёрно-белым).
            kernel (np.ndarray): Ядро свёртки (матрица).

        Returns:
            tuple: (result_image, execution_time)
        """
        start_time = time.time()
        
        image_height, image_width = image.shape[:2]
        kernel_height, kernel_width = kernel.shape
        
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2
        
        if len(image.shape) == 3:
            padded_image = np.pad(image, 
                                ((pad_height, pad_height), 
                                 (pad_width, pad_width), 
                                 (0, 0)), 
                                mode='constant')
        else:
            padded_image = np.pad(image, 
                                ((pad_height, pad_height), 
                                 (pad_width, pad_width)), 
                                mode='constant')
        
        output = np.zeros_like(image, dtype=np.float32)
        
        for i in range(image_height):
            for j in range(image_width):
                if len(image.shape) == 3:
                    for channel in range(3):
                        region = padded_image[i:i+kernel_height, j:j+kernel_width, channel]
                        output[i, j, channel] = np.sum(region * kernel)
                else:
                    region = padded_image[i:i+kernel_height, j:j+kernel_width]
                    output[i, j] = np.sum(region * kernel)
        
        output = np.clip(output, 0, 255).astype(np.uint8)
    
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Свёртка выполнена за {execution_time:.4f} секунд")
    
        return output, execution_time

    def rgb_to_grayscale(self, image: np.ndarray) -> tuple:
        start_time = time.time()
        
        grayscale = np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.float32)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Преобразование в grayscale выполнено за {execution_time:.4f} секунд")
        return grayscale, execution_time

    def gamma_correction(self, image: np.ndarray, gamma: float = 1.0) -> tuple:
        """
        Применяет гамма-коррекцию к изображению.

        Args:
            image (np.ndarray): Входное изображение.
            gamma (float): Коэффициент гамма-коррекции (>0).

        Returns:
            tuple: (corrected_image, execution_time)
        """
        start_time = time.time()
        
        normalized_image = image.astype(np.float32) / 255.0
        corrected_image = np.power(normalized_image, gamma)
        result = (corrected_image * 255).astype(np.uint8)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Гамма-коррекция выполнена за {execution_time:.4f} секунд")
        return result, execution_time

    def _sobel_operator(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Применяет оператор Собеля для вычисления градиентов.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            tuple: (gradient_x, gradient_y)
        """
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
    
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
    
        gradient_x = self.convolution(image, sobel_x)[0]
        gradient_y = self.convolution(image, sobel_y)[0]
    
        return gradient_x, gradient_y

    def simple_blur(self, img: np.ndarray) -> np.ndarray:
        blur_kernel = np.array([[1/9, 1/9, 1/9],
                            [1/9, 1/9, 1/9],
                            [1/9, 1/9, 1/9]])
        
        result, _ = self.convolution(img, blur_kernel)
        return result

    def edge_detection(self, image: np.ndarray) -> tuple:
        """
        Выполняет обнаружение границ на изображении.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            tuple: (edges_image, execution_time)
        """
        start_time = time.time()
    
        # Преобразование в оттенки серого
        gray, _ = self.rgb_to_grayscale(image)
        
        # Оператор Собеля
        gradient_x, gradient_y = self._sobel_operator(gray)
        
        # Магнитуда градиента
        gradient_magnitude = np.sqrt(gradient_x.astype(np.float32)**2 + 
                                    gradient_y.astype(np.float32)**2)
        
        # Нормализация
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
        
        # Пороговая обработка
        edges = np.where(gradient_magnitude > 50, 255, 0).astype(np.uint8)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return edges, execution_time

    def corner_detection(self, image: np.ndarray) -> tuple:
        """
        Выполняет обнаружение углов на изображении.
        """
        start_time = time.time()
        
        try:
            if len(image.shape) == 3:
                gray, _ = self.rgb_to_grayscale(image)
            else:
                gray = image.astype(np.float32)
            
            # Вычисление производных
            ix = np.zeros_like(gray)
            iy = np.zeros_like(gray)
            
            ix[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
            iy[1:-1, :] = gray[2:, :] - gray[:-2, :]
            
            
            ix2 = self.simple_blur(ix * ix)
            iy2 = self.simple_blur(iy * iy)
            ixy = self.simple_blur(ix * iy)
            
            # Отклик Харриса
            det = ix2 * iy2 - ixy * ixy
            trace = ix2 + iy2
            harris_response = det - 0.04 * (trace ** 2)

            threshold = 0.1 * harris_response.max()
            corners_y, corners_x = np.where(harris_response > threshold)
            
            result = image.copy()
            for i in range(len(corners_x)):
                x, y = corners_x[i], corners_y[i]
                if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
                    cv2.circle(result, (int(x), int(y)), 2, (0, 0, 255), -1)
            
            print(f"Найдено углов: {len(corners_x)}")
            
        except Exception as e:
            print(f"Ошибка: {e}")
            result = image.copy()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return result, execution_time

    def circle_detection(self, image: np.ndarray) -> tuple:
        """
        Выполняет обнаружение окружностей на изображении с помощью преобразования Хафа.
        """
        start_time = time.time()
        
        try:
            # Уменьшение размера изображения для ускорения
            scale_factor = 0.3
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            # Простое уменьшение изображения (nearest neighbor)
            if len(image.shape) == 3:
                small_image = np.zeros((new_h, new_w, 3), dtype=image.dtype)
                for i in range(new_h):
                    for j in range(new_w):
                        orig_i = min(int(i / scale_factor), h-1)
                        orig_j = min(int(j / scale_factor), w-1)
                        small_image[i, j] = image[orig_i, orig_j]
            else:
                small_image = np.zeros((new_h, new_w), dtype=image.dtype)
                for i in range(new_h):
                    for j in range(new_w):
                        orig_i = min(int(i / scale_factor), h-1)
                        orig_j = min(int(j / scale_factor), w-1)
                        small_image[i, j] = image[orig_i, orig_j]
            
            # Преобразование в градации серого
            gray, _ = self.rgb_to_grayscale(small_image)
            
            # Обнаружение границ
            edges, _ = self.edge_detection(small_image)
            edges_binary = (edges > 100).astype(np.uint8) * 255
            
            # Параметры преобразования Хафа
            height, width = edges_binary.shape
            min_radius = 20
            max_radius = min(height, width) // 6
            radius_step = 2
            
            print(f"Диапазон радиусов: {min_radius}-{max_radius} с шагом {radius_step}")
            
            # Сэмплирование точек границ
            edge_points = np.argwhere(edges_binary > 0)
            sampling_rate = 2
            sampled_edge_points = edge_points[::sampling_rate]
            
            print(f"Всего точек границ: {len(edge_points)}, после сэмплирования: {len(sampled_edge_points)}")
            
            # Создаем аккумулятор
            accumulator = np.zeros((height, width, (max_radius - min_radius) // radius_step + 1), 
                                dtype=np.uint16)
            
            # Голосование
            print("Начало оптимизированного голосования...")
            
            for i, (y, x) in enumerate(sampled_edge_points):
                if i % 500 == 0:
                    print(f"Обработано {i}/{len(sampled_edge_points)} точек...")
                
                angle_step = 6
                for angle in range(0, 360, angle_step):
                    theta = np.radians(angle)
                    
                    for r_idx, radius in enumerate(range(min_radius, max_radius + 1, radius_step)):
                        a = int(x - radius * np.cos(theta))
                        b = int(y - radius * np.sin(theta))
                        
                        if 0 <= a < width and 0 <= b < height:
                            accumulator[b, a, r_idx] += 1
            
            print("Поиск окружностей...")
            
            threshold = 0.6*np.max(accumulator)
            circles = []
            
            for r_idx, radius in enumerate(range(min_radius, max_radius + 1, radius_step)):
                acc_layer = accumulator[:, :, r_idx]
                strong_candidates = np.argwhere(acc_layer > threshold)
                
                for y, x in strong_candidates:
                    is_duplicate = False
                    for existing_x, existing_y, existing_r in circles:
                        distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                        if distance < 38 and abs(radius - existing_r) < 25:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        circles.append((x, y, radius))
                        if len(circles) >= 20:
                            break
                
                if len(circles) >= 20:
                    break
            
            print(f"Найдено окружностей: {len(circles)}")
            
            # Масштабируем координаты обратно
            result = image.copy()
            for (x, y, radius) in circles:
                x_orig = int(x / scale_factor)
                y_orig = int(y / scale_factor)
                radius_orig = int(radius / scale_factor)
                
                self._draw_circle(result, x_orig, y_orig, radius_orig, (0, 255, 0), 2)
                cv2.circle(result, (x_orig, y_orig), 3, (0, 0, 255), -1)
                
        except Exception as e:
            print(f"Ошибка при обнаружении окружностей: {e}")
            result = image.copy()
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Обнаружение окружностей выполнено за {execution_time:.4f} секунд")
        
        return result, execution_time

    def _draw_circle(self, image: np.ndarray, center_x: int, center_y: int, 
                    radius: int, color: tuple, thickness: int = 2) -> None:
        """
        Рисует окружность на изображении с помощью алгоритма Брезенхэма.

        Args:
            image (np.ndarray): Изображение для рисования.
            center_x (int): X-координата центра.
            center_y (int): Y-координата центра.
            radius (int): Радиус окружности.
            color (tuple): Цвет в формате BGR.
            thickness (int): Толщина линии.
        """
        x = 0
        y = radius
        d = 3 - 2 * radius
        
        def draw_points(xc, yc, x, y):
            points = []
            points.extend([
                (xc + x, yc + y), (xc - x, yc + y),
                (xc + x, yc - y), (xc - x, yc - y),
                (xc + y, yc + x), (xc - y, yc + x), 
                (xc + y, yc - x), (xc - y, yc - x)
            ])
            
            if thickness > 1:
                for dx in range(-thickness//2, thickness//2 + 1):
                    for dy in range(-thickness//2, thickness//2 + 1):
                        if dx != 0 or dy != 0:
                            for px, py in points.copy():
                                points.append((px + dx, py + dy))
            
            for px, py in points:
                if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                    image[py, px] = color
        
        while y >= x:
            draw_points(center_x, center_y, x, y)
            
            x += 1
            if d > 0:
                y -= 1
                d = d + 4 * (x - y) + 10
            else:
                d = d + 4 * x + 6
            
            draw_points(center_x, center_y, x, y)