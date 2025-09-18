"""
Скрипт для запуска GUI приложения TrOCR
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Проверка зависимостей"""
    try:
        import PyQt5
        import qdarkstyle
        import matplotlib
        import torch
        import transformers
        print("✓ Все зависимости установлены")
        return True
    except ImportError as e:
        print(f"❌ Отсутствует зависимость: {e}")
        print("Установите зависимости: pip install -r requirements.txt")
        return False

def main():
    """Главная функция"""
    print("🚀 Запуск TrOCR GUI приложения...")
    
    # Проверяем зависимости
    if not check_dependencies():
        return 1
    
    # Проверяем наличие основного файла
    if not Path("trocr_gui.py").exists():
        print("❌ Файл trocr_gui.py не найден!")
        return 1
    
    # Импортируем и запускаем GUI
    try:
        from trocr_gui import main as gui_main
        gui_main()
    except Exception as e:
        print(f"❌ Ошибка при запуске GUI: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
