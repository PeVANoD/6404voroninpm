"""
main.py

Общий главный модуль для запуска лабораторных работ.

Запуск:
    python main.py lab1 <аргументы>    # Для лабораторной работы 1
    python main.py lab2 <аргументы>    # Для лабораторной работы 2

Пример:
    python main.py lab1 edges image.jpg
    python main.py lab2 --limit 5
"""

import sys
import subprocess


def main() -> None:
    """
    Главная функция для выбора лабораторной работы.
    """
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python main.py lab1 <аргументы>    # Лабораторная работа 1")
        print("  python main.py lab2 <аргументы>    # Лабораторная работа 2")
        print("\nПримеры:")
        print("  python main.py lab1 edges image.jpg")
        print("  python main.py lab2 --limit 5")
        return

    lab_number = sys.argv[1]
    
    if lab_number == "lab1":
        # Запуск лабораторной работы 1 с переданными аргументами
        from main_lab1 import main_lab1
        # Создаем аргументы для main_lab1
        lab1_args = sys.argv[2:]
        sys.argv = [sys.argv[0]] + lab1_args
        main_lab1()
        
    elif lab_number == "lab2":
        # Запуск лабораторной работы 2 с переданными аргументами
        from main_lab2 import main_lab2
        # Создаем аргументы для main_lab2
        lab2_args = sys.argv[2:]
        sys.argv = [sys.argv[0]] + lab2_args
        main_lab2()
        
    else:
        print("Ошибка: неизвестная лабораторная работа")
        print("Доступные варианты: lab1, lab2")


if __name__ == "__main__":
    main()