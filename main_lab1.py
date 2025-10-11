"""
main_lab1.py

Лабораторная работа №1 - Обработка изображений с помощью OpenCV.

Модуль предназначен для демонстрации работы с обработкой изображений с помощью библиотеки OpenCV.
Реализован консольный интерфейс для применения различных методов обработки к изображению.

Запуск:
    python main_lab1.py <метод> <путь_к_изображению> [-o путь_для_сохранения]

Аргументы:
    метод: convolution | grayscale | gamma | edges | corners | circles
    путь_к_изображению: путь к входному изображению
    -o, --output: путь для сохранения результата

Пример:
    python main_lab1.py edges input.jpg
    python main_lab1.py corners input.jpg -o corners_result.png
"""

import argparse
import os
import numpy as np
import cv2

from implementation import ImageProcessing


def main_lab1() -> None:
    """
    Главная функция для лабораторной работы №1.
    """
    parser = argparse.ArgumentParser(
        description="Обработка изображения с помощью методов ImageProcessing (OpenCV).",
    )
    parser.add_argument(
        "method",
        choices=[
            "convolution",
            "grayscale", 
            "gamma",
            "edges",
            "corners",
            "circles",
        ],
        help="Метод обработки изображения",
    )
    parser.add_argument(
        "input",
        help="Путь к входному изображению",
    )
    parser.add_argument(
        "-o", "--output",
        help="Путь для сохранения результата (по умолчанию: <input>_result.png)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Значение гамма-коррекции (только для метода gamma)",
    )

    args = parser.parse_args()

    # Загрузка изображения
    image = cv2.imread(args.input)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {args.input}")
        return

    processor = ImageProcessing()

    try:
        # Применение выбранного метода
        if args.method == "convolution":
            kernel = np.ones((5, 5)) / 25
            result_image, exec_time = processor.convolution(image, kernel)
        elif args.method == "grayscale":
            result_image, exec_time = processor.rgb_to_grayscale(image)
        elif args.method == "gamma":
            result_image, exec_time = processor.gamma_correction(image, args.gamma)
        elif args.method == "edges":
            result_image, exec_time = processor.edge_detection(image)
        elif args.method == "corners":
            result_image, exec_time = processor.corner_detection(image)
        elif args.method == "circles":
            result_image, exec_time = processor.circle_detection(image)
        else:
            print("Ошибка: неизвестный метод")
            return
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return

    # Сохранение результата
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        time_str = f"{exec_time:.2f}".replace('.', ',')
        output_path = f"{base}_{args.method}_result_{time_str}sec.png"

    cv2.imwrite(output_path, result_image)
    print(f"Результат сохранён в {output_path}")


if __name__ == "__main__":
    main_lab1()