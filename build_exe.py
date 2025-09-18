"""
Скрипт для создания исполняемого файла (.exe) для Windows
"""

import os
import sys
import subprocess
from pathlib import Path

def install_pyinstaller():
    """Установка PyInstaller"""
    try:
        import PyInstaller
        print("✓ PyInstaller уже установлен")
        return True
    except ImportError:
        print("Установка PyInstaller...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            print("✓ PyInstaller установлен")
            return True
        except subprocess.CalledProcessError:
            print("❌ Ошибка установки PyInstaller")
            return False

def create_spec_file():
    """Создание .spec файла для PyInstaller"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['trocr_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('GUI_README.md', '.'),
    ],
    hiddenimports=[
        'PyQt5.QtCore',
        'PyQt5.QtGui', 
        'PyQt5.QtWidgets',
        'matplotlib.backends.backend_qt5agg',
        'qdarkstyle',
        'torch',
        'transformers',
        'jiwer',
        'PIL',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'tqdm',
        'sklearn',
        'cv2'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TrOCR_GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None
)
'''
    
    with open('trocr_gui.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)
    
    print("✓ Создан файл trocr_gui.spec")

def build_executable():
    """Сборка исполняемого файла"""
    print("Начинается сборка исполняемого файла...")
    print("Это может занять несколько минут...")
    
    try:
        # Сборка с помощью PyInstaller
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--windowed",
            "--name=TrOCR_GUI",
            "--add-data=GUI_README.md;.",
            "--hidden-import=PyQt5.QtCore",
            "--hidden-import=PyQt5.QtGui",
            "--hidden-import=PyQt5.QtWidgets",
            "--hidden-import=matplotlib.backends.backend_qt5agg",
            "--hidden-import=qdarkstyle",
            "--hidden-import=torch",
            "--hidden-import=transformers",
            "--hidden-import=jiwer",
            "--hidden-import=PIL",
            "--hidden-import=numpy",
            "--hidden-import=pandas",
            "--hidden-import=matplotlib",
            "--hidden-import=seaborn",
            "--hidden-import=tqdm",
            "--hidden-import=sklearn",
            "--hidden-import=cv2",
            "trocr_gui.py"
        ]
        
        subprocess.check_call(cmd)
        print("✓ Исполняемый файл создан успешно!")
        print("📁 Файл находится в папке 'dist/TrOCR_GUI.exe'")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при сборке: {e}")
        return False

def main():
    """Главная функция"""
    print("🔨 Создание исполняемого файла TrOCR GUI")
    print("=" * 50)
    
    # Проверяем наличие основного файла
    if not Path("trocr_gui.py").exists():
        print("❌ Файл trocr_gui.py не найден!")
        return 1
    
    # Устанавливаем PyInstaller
    if not install_pyinstaller():
        return 1
    
    # Создаем .spec файл
    create_spec_file()
    
    # Собираем исполняемый файл
    if build_executable():
        print("\n🎉 Готово!")
        print("Теперь вы можете запустить TrOCR_GUI.exe из папки dist/")
        print("Размер файла может быть большим из-за включенных ML библиотек")
        return 0
    else:
        print("\n❌ Ошибка при создании исполняемого файла")
        return 1

if __name__ == "__main__":
    sys.exit(main())
