"""
main_lab4.py

Лабораторная работа №4 - Асинхронная обработка изображений с параллельными процессами.
"""

import argparse
import asyncio
import time
import sys
from async_cat_image_processor import AsyncCatImageProcessor


async def main_lab4_async(args_list=None) -> None:
    """
    Главная асинхронная функция для лабораторной работы №4.
    """
    parser = argparse.ArgumentParser(
        description="Асинхронная загрузка и обработка изображений животных через API.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Количество изображений для загрузки (по умолчанию: 3)",
    )
    parser.add_argument(
        "--output_dir",
        default="processed_images_async",
        help="Директория для сохранения результатов (по умолчанию: processed_images_async)",
    )

    # Парсим аргументы
    if args_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_list)

    try:
        print("🚀 Запуск лабораторной работы №4")
        print(f"📊 Будет загружено изображений: {args.limit}")
        print(f"📁 Директория для сохранения: {args.output_dir}")
        print("=" * 50)
        
        start_time = time.time()
        
        # Создаем асинхронный процессор
        processor = AsyncCatImageProcessor(limit=args.limit, output_dir=args.output_dir)
        
        # Запускаем асинхронный генераторный пайплайн (заменили process_images_async() на run_pipeline())
        await processor.run_pipeline()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("=" * 50)
        print(f"🎉 Асинхронная обработка завершена!")
        print(f"✅ Обработано изображений: {len(processor.downloaded_images)}")
        print(f"⏱️ Общее время выполнения: {total_time:.2f} секунд")
        print(f"💾 Результаты сохранены в директории: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()


def main_lab4(args_list=None):
    """
    Синхронная обертка для запуска асинхронной функции.
    """
    if args_list is None:
        asyncio.run(main_lab4_async())
    else:
        asyncio.run(main_lab4_async(args_list))


if __name__ == "__main__":
    main_lab4()