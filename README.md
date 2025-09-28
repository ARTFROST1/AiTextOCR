# AiTextOCR - TrOCR Evaluation Project

## ⚡ Быстрый запуск (для новичков)

**Только что клонировали проект? Выполните эти 4 шага:**

1. **Откройте терминал в папке проекта**
2. **Установите окружение:** `python scripts/setup_environment.py`
3. **Активируйте окружение:** `./venv_cuda/Scripts/Activate.ps1` (Windows) или `source venv_cuda/bin/activate` (Linux/macOS)
4. **Запустите GUI:** `python run_gui.py`

**Готово!** 🎉 Приложение запущено и готово к работе.

---

## 📋 Описание проекта

Система оценки модели **TrOCR (Transformer-based OCR)** на датасете **IAM** для распознавания рукописного текста с современным GUI интерфейсом.

**Основные метрики:**
- **WER (Word Error Rate)** - процент ошибок на уровне слов
- **CER (Character Error Rate)** - процент ошибок на уровне символов  
- **Accuracy** - точность распознавания

## 🚀 Быстрая установка и запуск

### ✅ Системные требования
- **Python 3.12.10** (обязательно)
- **RAM**: минимум 8GB (рекомендуется 16GB)
- **Место на диске**: ~5GB
- **GPU**: не обязательно, но ускорит работу
- **ОС**: Windows, Linux, macOS

### 📦 Установка после клонирования

**1. Клонируйте репозиторий:**
```bash
git clone <URL_РЕПОЗИТОРИЯ>
cd AiTextOCR
```

**2. Автоматическая установка окружения:**

**Windows (PowerShell):**
```powershell
./setup.ps1
```

**Windows (CMD):**
```bat
setup.bat
```

**Универсальный способ (все ОС):**
```bash
python scripts/setup_environment.py
```

**3. Активируйте виртуальное окружение:**

**Windows:**
```powershell
.\venv_cuda\Scripts\activate
```

**Linux/macOS:**
```bash
source venv_cuda/bin/activate
```

**4. Запустите приложение:**

**GUI (рекомендуется):**
```bash
python run_gui.py
```

**Консольный режим:**
```bash
python scripts/run_full_evaluation.py
```

### ⚙️ Что делает скрипт установки

Скрипт `scripts/setup_environment.py` автоматически:
- ✅ Создаёт виртуальное окружение `venv_cuda/`
- ✅ Устанавливает PyTorch с CUDA (если доступно) или CPU версию
- ✅ Устанавливает все зависимости из `requirements.txt`
- ✅ Проверяет корректность установки
- ✅ Создаёт необходимые папки (`results/`, `profiles/`)

**Принудительное отключение GPU:**
```bash
# Windows PowerShell
$env:FORCE_CPU="1"; python scripts/setup_environment.py

# Linux/macOS
FORCE_CPU=1 python scripts/setup_environment.py
```

### 🔧 Если автоустановка не работает

**Проблема:** Скрипт установки завершается с ошибками или зависимости не работают.

**Решение - ручная установка:**

1. **Создайте виртуальное окружение:**
```bash
python -m venv venv_cuda
```

2. **Активируйте окружение:**
```bash
# Windows PowerShell
.\venv_cuda\Scripts\Activate.ps1

# Windows CMD  
.\venv_cuda\Scripts\activate.bat

# Linux/macOS
source venv_cuda/bin/activate
```

3. **Обновите pip:**
```bash
pip install --upgrade pip setuptools wheel
```

4. **Установите PyTorch:**
```bash
# Для GPU (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Для CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

5. **Установите остальные зависимости по одной:**
```bash
pip install transformers==4.41.2
pip install datasets==2.21.0
pip install Pillow==10.3.0
pip install numpy==1.26.4
pip install matplotlib==3.8.4
pip install seaborn==0.13.2
pip install tqdm==4.66.4
pip install scikit-learn==1.4.2
pip install jiwer==3.0.4
pip install opencv-python==4.9.0.80
pip install pandas==2.2.2
pip install PyQt5==5.15.10
pip install qdarkstyle==3.2.3
pip install easyocr==1.7.1
pip install scikit-image==0.22.0
```

**Возможные причины проблем:**
- Устаревший pip или setuptools
- Конфликты версий пакетов
- Проблемы с сетевым подключением
- Ограничения корпоративного файрвола
- Недостаточно прав доступа

## 🏗️ Структура проекта

```
📦 AiTextOCR/
├── 📄 README.md                    # Главная документация (этот файл)
├── 📄 requirements.txt             # Python зависимости
├── 📄 setup.ps1 / setup.bat       # Скрипты установки для Windows
├── 📄 run_gui.py                   # Запуск GUI приложения
├── 📄 trocr_gui.py                 # Основной GUI код
├── 📄 trocr_evaluation.py          # Логика оценки TrOCR
├── 📄 easyocr_evaluation.py        # Логика оценки EasyOCR
├── 📄 build_exe.py                 # Сборка исполняемого файла
├── 📁 scripts/                     # 🔧 Утилиты и скрипты
│   ├── setup_environment.py        # Автоматическая настройка окружения
│   ├── download_real_iam.py        # Загрузка датасета IAM
│   └── run_full_evaluation.py      # Консольная оценка
├── 📁 venv_cuda/                   # 🐍 Виртуальное окружение (создается автоматически)
├── 📁 results/                     # 📈 Результаты тестов (создается автоматически)
├── 📁 profiles/                    # 💾 Профили настроек GUI
├── 📁 datasets/                    # 📊 Тестовые датасеты
└── 📁 IAM/                         # 📊 Датасет IAM (загружается автоматически)
```

## ✨ Возможности приложения

### 🎨 Современный GUI
- **Тёмная тема** с современным дизайном
- **Консольное окно** с подсветкой синтаксиса
- **Система профилей** для сохранения настроек
- **Прогресс-бар** с анимациями
- **Графики и таблицы** результатов

### 🔧 Функциональность
- **Оценка TrOCR моделей** на различных датасетах
- **Поддержка EasyOCR** для сравнения
- **Автоматическое определение GPU/CPU**
- **Экспорт результатов** в JSON, CSV, PNG
- **Настраиваемые параметры** оценки

## 📊 Использование

### 🖥️ GUI режим (рекомендуется)

После установки запустите:
```bash
python run_gui.py
```

**Основные функции GUI:**
1. **Выбор модели** - TrOCR base/large, handwritten/printed
2. **Выбор датасета** - IAM, пользовательские изображения
3. **Настройка параметров** - количество изображений, пути сохранения
4. **Мониторинг прогресса** - в реальном времени
5. **Просмотр результатов** - таблицы, графики, статистика

### 📋 Консольный режим

**Полная автоматическая оценка:**
```bash
python scripts/run_full_evaluation.py
```

**Загрузка датасета IAM:**
```bash
python scripts/download_real_iam.py
```

### 📈 Результаты

Все результаты сохраняются в папке `results/YYYY-MM-DD_HH-MM-SS_model/`:
- **`evaluation_results.json`** - детальные результаты с метриками
- **`evaluation_results.csv`** - таблица для анализа в Excel/Python
- **`evaluation_plots.png`** - графики распределения метрик

**Основные метрики:**
- **WER (Word Error Rate)** - ошибки на уровне слов
- **CER (Character Error Rate)** - ошибки на уровне символов
- **Accuracy** - общая точность распознавания

## 🔧 Настройка и конфигурация

### 🎯 Смена модели TrOCR

Доступные модели:
- `microsoft/trocr-base-handwritten` (по умолчанию)
- `microsoft/trocr-large-handwritten` (больше, точнее)
- `microsoft/trocr-base-printed` (для печатного текста)
- `microsoft/trocr-large-printed` (большая, печатная)

### ⚙️ Настройка параметров

В GUI или в файлах скриптов можно настроить:
- Количество обрабатываемых изображений
- Пути к датасетам
- Параметры сохранения результатов
- Использование GPU/CPU

## 🚨 Устранение проблем

### ❌ Ошибка "модуль не найден"
```bash
# Убедитесь что виртуальное окружение активировано
.\venv_cuda\Scripts\activate  # Windows
source venv_cuda/bin/activate  # Linux/macOS

# Переустановите зависимости
pip install -r requirements.txt
```

### 💾 Недостаточно памяти
- Закройте другие программы
- Используйте модель `trocr-base` вместо `trocr-large`
- Уменьшите количество обрабатываемых изображений

### 🐌 Медленная работа
- Убедитесь что используется GPU (если доступно)
- Уменьшите размер датасета для тестирования
- Используйте более быструю модель

### 🔧 Проблемы с установкой
```bash
# Принудительная переустановка PyTorch
pip uninstall torch torchvision
python scripts/setup_environment.py

# Проверка установки
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## 📄 Лицензия и благодарности

**Используемые технологии:**
- **TrOCR** - Microsoft (MIT License)
- **PyTorch** - Facebook AI Research
- **Transformers** - Hugging Face
- **PyQt5** - GUI фреймворк
- **IAM Dataset** - для исследовательских целей

---

**🎯 Готово к использованию!** После выполнения установки запустите `python run_gui.py` и начните оценку OCR моделей.
