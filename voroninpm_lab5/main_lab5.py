"""
main_lab5.py

Лабораторная работа №5 - Асинхронная обработка изображений с асинхронным логированием.
"""

import argparse
import asyncio
import time
import sys
from .async_cat_image_processor_v2 import AsyncCatImageProcessorV2
from .async_logging_config import setup_async_logging
from .config import DEFAULT_OUTPUT_DIR_V2


async def main_lab5_async(args_list=None) -> None:
    """
    Главная асинхронная функция для лабораторной работы №5.
    """
    parser = argparse.ArgumentParser(
        description="Асинхронная загрузка и обработка изображений животных с асинхронным логированием.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python -m 6404voroninpm_lab5 --limit 5
  python -m 6404voroninpm_lab5 --limit 3 --output-dir my_images
        """
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Количество изображений для загрузки (по умолчанию: 3)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR_V2,
        help=f"Директория для сохранения результатов (по умолчанию: {DEFAULT_OUTPUT_DIR_V2})",
    )

    # Парсим аргументы
    if args_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_list)

    try:
        # Настраиваем асинхронное логирование
        setup_async_logging()
        
        start_time = time.time()
        
        # Создаем асинхронный процессор версии 2
        processor = AsyncCatImageProcessorV2(limit=args.limit, output_dir=args.output_dir)
        
        # Запускаем асинхронный генераторный пайплайн
        await processor.run_pipeline()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"💾 Результаты сохранены в директории: {args.output_dir}")
        print(f"⏱️ Общее время выполнения: {total_time:.2f} секунд")
        
    except KeyboardInterrupt:
        print("\n⚠️ Программа прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")
        sys.exit(1)


def main_lab5(args_list=None):
    """
    Синхронная обертка для запуска асинхронной функции lab5.
    """
    if args_list is None:
        asyncio.run(main_lab5_async())
    else:
        asyncio.run(main_lab5_async(args_list))


if __name__ == "__main__":
    main_lab5()