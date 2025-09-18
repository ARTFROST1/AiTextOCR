"""
Скрипт для автоматической настройки окружения (CUDA-ready)
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Выполнение команды с выводом описания"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} завершено успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Ошибка при {description.lower()}: {e}")
        print(f"Вывод ошибки: {e.stderr}")
        return False


def check_python_version():
    """Проверка версии Python"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 12):
        print("✗ Требуется Python 3.12 или выше для CUDA-сборок PyTorch")
        print(f"Текущая версия: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✓ Python версия: {version.major}.{version.minor}.{version.micro}")
        return True


def create_virtual_environment():
    """Создание виртуального окружения venv_cuda"""
    venv_path = Path("venv_cuda")

    if venv_path.exists():
        print("✓ Виртуальное окружение venv_cuda уже существует")
        return True

    return run_command(f"{sys.executable} -m venv venv_cuda", "Создание виртуального окружения venv_cuda")


def get_pip_command():
    """Получение команды для pip в зависимости от ОС"""
    if os.name == 'nt':  # Windows
        return "venv_cuda\\Scripts\\pip"
    else:  # Linux/macOS
        return "venv_cuda/bin/pip"


def get_python_command():
    if os.name == 'nt':
        return "venv_cuda\\Scripts\\python"
    else:
        return "venv_cuda/bin/python"


def install_pytorch_cuda():
    """Установка CUDA-сборки PyTorch (cu121)"""
    pip_cmd = get_pip_command()
    # Сначала обновим pip
    if not run_command(f"{pip_cmd} install --upgrade pip", "Обновление pip"):
        return False
    # Затем ставим CUDA-сборки torch/torchvision
    return run_command(
        f"{pip_cmd} install --index-url https://download.pytorch.org/whl/cu121 torch torchvision",
        "Установка PyTorch CUDA (cu121)"
    )


def install_requirements():
    """Установка остальных зависимостей"""
    pip_cmd = get_pip_command()
    return run_command(f"{pip_cmd} install -r requirements.txt", "Установка зависимостей из requirements.txt")


def test_installation():
    """Тестирование установки"""
    print("\nТестирование установки...")

    try:
        import torch
        import transformers
        import jiwer
        import PIL
        import numpy as np
        import pandas as pd
        import matplotlib
        import seaborn
        import tqdm
        import sklearn
        import cv2

        print("✓ Все основные библиотеки установлены успешно")

        # Проверяем доступность GPU
        if torch.cuda.is_available():
            print(f"✓ CUDA доступна: {torch.cuda.get_device_name(0)} (cuda {torch.version.cuda})")
        else:
            print("ℹ CUDA недоступна, будет использоваться CPU")

        return True

    except ImportError as e:
        print(f"✗ Ошибка импорта: {e}")
        return False


def create_project_structure():
    """Создание структуры проекта"""
    print("\nСоздание структуры проекта...")

    directories = [
        "results"
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Создана директория: {directory}")

    return True


def main():
    """Основная функция настройки"""
    print("="*60)
    print("НАСТРОЙКА CUDA ОКРУЖЕНИЯ ДЛЯ TrOCR EVALUATION")
    print("="*60)

    if not check_python_version():
        sys.exit(1)

    if not create_virtual_environment():
        print("\nПопробуйте создать виртуальное окружение вручную:")
        print(f"{sys.executable} -m venv venv_cuda")
        sys.exit(1)

    if not install_pytorch_cuda():
        print("\nНе удалось установить CUDA PyTorch. Проверьте совместимость CUDA/драйверов.")
        sys.exit(1)

    if not install_requirements():
        print("\nПопробуйте установить зависимости вручную:")
        print(get_python_command())
        print("-m pip install -r requirements.txt")
        sys.exit(1)

    if not test_installation():
        print("\nЕсть проблемы с установкой. Проверьте зависимости вручную.")
        sys.exit(1)

    create_project_structure()

    print("\n" + "="*60)
    print("НАСТРОЙКА ЗАВЕРШЕНА УСПЕШНО!")
    print("="*60)

    print("\nСледующие шаги:")
    print("1. Активируйте виртуальное окружение:")
    if os.name == 'nt':
        print("   venv_cuda\\Scripts\\activate")
    else:
        print("   source venv_cuda/bin/activate")

    print("\n2. Запустите оценку:")
    print("   python scripts/run_full_evaluation.py")


if __name__ == "__main__":
    main()
