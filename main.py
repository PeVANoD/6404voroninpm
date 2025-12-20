"""
main.py

Общий главный модуль для запуска лабораторных работ.
"""

import sys
import subprocess
import asyncio


def main() -> None:
    """
    Главная функция для выбора лабораторной работы.
    """
    if len(sys.argv) < 2:
        print_help()
        return

    lab_number = sys.argv[1]
    
    if lab_number == "lab1":
        from main_lab1 import main_lab1
        lab1_args = sys.argv[2:]
        sys.argv = [sys.argv[0]] + lab1_args
        main_lab1()
        
    elif lab_number == "lab2":
        from main_lab2 import main_lab2
        lab2_args = sys.argv[2:]
        sys.argv = [sys.argv[0]] + lab2_args
        main_lab2()

    elif lab_number == "lab3":
        from airlines.main3 import main_lab3
        show_plots = "-show" in sys.argv
        main_lab3(show_plots=show_plots)

    elif lab_number == "lab4":
        from main_lab4 import main_lab4
        lab4_args = sys.argv[2:]
        main_lab4(lab4_args)

    elif lab_number == "lab5":
        from voroninpm_lab5 import main_lab5
        main_lab5(sys.argv[2:])

    elif lab_number == "test":
        run_tests()
        
    else:
        print(f"Ошибка: неизвестная лабораторная работа '{lab_number}'")
        print_help()


def print_help():
    """Вывод справки по использованию."""
    print("Использование: python main.py <команда> [аргументы]")
    print("\nКоманды:")
    print("  lab1 <аргументы>    # Лабораторная работа 1 - Обработка изображений")
    print("  lab2 <аргументы>    # Лабораторная работа 2 - Анализ данных")
    print("  lab3 [-show]        # Лабораторная работа 3 - Визуализация")
    print("  lab4 <аргументы>    # Лабораторная работа 4 - Базовая асинхронная обработка")
    print("  lab5 <аргументы>    # Лабораторная работа 5 - Асинхронная обработка с логированием")
    print("  test                # Запуск тестов")


def run_tests():
    """Запуск тестов для лабораторных работ."""
    import unittest
    import sys
    import os
    
    print("🧪 Запуск тестов...")
    
    try:
        # Добавляем текущую директорию в путь
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Импортируем тесты
        from tests.test_cat_image import TestCatImage, TestAsyncLoggingIntegration
        from tests.test_processor_v2 import TestAsyncProcessorV2, TestAsyncProcessorV2Sync
        from tests.test_async_logging import TestAsyncLogging
        
        # Создаем тестовый набор
        loader = unittest.TestLoader()
        
        # Создаем тестовую suite
        test_suite = unittest.TestSuite()
        
        # Добавляем тесты для lab4
        test_suite.addTests(loader.loadTestsFromTestCase(TestCatImage))
        
        # Добавляем тесты для lab5
        test_suite.addTests(loader.loadTestsFromTestCase(TestAsyncLoggingIntegration))
        test_suite.addTests(loader.loadTestsFromTestCase(TestAsyncProcessorV2))
        test_suite.addTests(loader.loadTestsFromTestCase(TestAsyncProcessorV2Sync))
        test_suite.addTests(loader.loadTestsFromTestCase(TestAsyncLogging))
        
        # Запускаем тесты
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        
        # Выводим итог
        print(f"\n📊 Результаты тестов:")
        print(f"Всего тестов: {result.testsRun}")
        print(f"Успешно: {result.testsRun - len(result.failures) - len(result.errors)}")
        print(f"Провалено: {len(result.failures)}")
        print(f"Ошибок: {len(result.errors)}")
        
        # Возвращаем код ошибки, если тесты не прошли
        if not result.wasSuccessful():
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Ошибка при запуске тестов: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()