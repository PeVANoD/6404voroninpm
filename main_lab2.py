"""
main_lab2.py

Лабораторная работа №2 - Работа с API изображений животных.

Модуль предназначен для загрузки изображений животных через API и их обработки.
Реализовано обнаружение контуров пользовательскими и библиотечными методами.

Запуск:
    python main_lab2.py [--limit N] [--output_dir путь]

Аргументы:
    --limit: количество изображений для загрузки (по умолчанию: 3)
    --output_dir: директория для сохранения результатов (по умолчанию: processed_images)

Пример:
    python main_lab2.py --limit 5
    python main_lab2.py --output_dir my_results --limit 2
"""

import argparse
from cat_image_processor import CatImageProcessor


def main_lab2() -> None:
    """
    Главная функция для лабораторной работы №2.
    """
    parser = argparse.ArgumentParser(
        description="Загрузка и обработка изображений животных через API.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Количество изображений для загрузки (по умолчанию: 3)",
    )
    parser.add_argument(
        "--output_dir",
        default="processed_images",
        help="Директория для сохранения результатов (по умолчанию: processed_images)",
    )

    args = parser.parse_args()

    try:
        print("Запуск лабораторной работы №2")
        print(f"Будет загружено изображений: {args.limit}")
        print(f"Директория для сохранения: {args.output_dir}")
        
        # Создаем процессор для обработки изображений
        processor = CatImageProcessor(limit=args.limit, output_dir=args.output_dir)
        
        # Загружаем и обрабатываем изображения
        processor.process_images()
        
        print(f"\nОбработка завершена!")
        print(f"Обработано изображений: {len(processor.downloaded_images)}")
        print(f"Результаты сохранены в директории: {args.output_dir}")
        
    except Exception as e:
        print(f"✗ Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_lab2()