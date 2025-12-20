"""
__main__.py

Точка входа для запуска пакета через python -m 6404voroninpm_lab5
"""

import sys
from .main_lab5 import main_lab5

def main():
    """Основная функция для запуска из командной строки."""
    # Передаем аргументы командной строки (кроме имени скрипта)
    main_lab5(sys.argv[1:])

if __name__ == "__main__":
    main()