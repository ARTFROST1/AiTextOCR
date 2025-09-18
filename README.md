# TrOCR Evaluation на датасете IAM

## 📋 Описание проекта

Этот проект реализует систему оценки модели **TrOCR (Transformer-based OCR)** на датасете **IAM (Institute of Automation and Mathematics)** для распознавания рукописного текста. Система вычисляет основные метрики качества OCR:

- **WER (Word Error Rate)** - процент ошибок на уровне слов
- **CER (Character Error Rate)** - процент ошибок на уровне символов  
- **Accuracy** - точность распознавания

## 🏗️ Структура проекта

```
📦 AiTextOCR - TrOCR Evaluation Project
├── 📄 README.md                  # Главная документация
├── 📄 requirements.txt           # Зависимости Python
├── 📄 trocr_evaluation.py        # Основная логика оценки
├── 📄 trocr_gui.py               # GUI приложение
├── 📄 run_gui.py                 # Запуск GUI
├── 📄 build_exe.py               # Сборка исполняемого файла
├── 📄 test_improved_metrics.py   # Тесты метрик
├── 📁 docs/                      # 📚 Вся документация
│   ├── PROJECT_GUIDE.md          # Руководство по проекту
│   ├── PROJECT_STRUCTURE.md      # Структура проекта
│   ├── QUICK_START.md            # Быстрый старт
│   ├── USAGE_EXAMPLES.md         # Примеры использования
│   ├── GUI_README.md             # Документация GUI
│   ├── CONSOLE_OUTPUT_GUIDE.md   # Консольный вывод
│   ├── RESULTS_FOLDER_SYSTEM.md  # Система папок результатов
│   ├── UPDATED_BUTTONS_GUIDE.md  # Обновленные кнопки
│   ├── METRICS_IMPROVEMENTS.md   # Улучшения метрик
│   ├── FINAL_*.md                # Финальные сводки
│   ├── REAL_DATASET_REPORT.md    # Отчет по датасету
│   └── FINAL_SUMMARY.md          # Итоговый отчет
├── 📁 scripts/                   # 🔧 Скрипты
│   ├── setup_environment.py      # Настройка окружения
│   ├── download_real_iam.py      # Загрузка датасета IAM
│   └── run_full_evaluation.py    # Запуск полной оценки
├── 📁 IAM/                       # 📊 Датасет IAM
│   ├── gt_test.txt               # Аннотации
│   └── image/                    # Изображения рукописного текста
├── 📁 results/                   # 📈 Результаты оценки
│   └── YYYY-MM-DD_HH-MM-SS_model/ # Папки с результатами тестов
│       ├── evaluation_results.json
│       ├── evaluation_results.csv
│       └── evaluation_plots.png
└── 📁 venv_cuda/                 # 🐍 Виртуальное окружение
```

## 🚀 Быстрый старт

### 🖥️ GUI приложение (Рекомендуется)
```bash
# Запуск графического интерфейса
python run_gui.py
```

### 📋 Консольный режим

#### 1. Настройка окружения
```bash
python scripts/setup_environment.py
```

#### 2. Загрузка датасета IAM
```bash
python scripts/download_real_iam.py
```

#### 3. Запуск оценки модели
```bash
python scripts/run_full_evaluation.py
```

### 📊 Просмотр результатов
Результаты сохраняются в папке `results/YYYY-MM-DD_HH-MM-SS_model/`:
- `evaluation_results.json` - детальные результаты
- `evaluation_results.csv` - таблица для анализа
- `evaluation_plots.png` - графики метрик

## 📊 Результаты оценки

### Последние результаты на датасете IAM:

| Метрика | Значение | Описание |
|---------|----------|----------|
| **WER** | 104.00% ± 39.80% | Процент ошибок на уровне слов |
| **CER** | 95.86% ± 30.26% | Процент ошибок на уровне символов |
| **Accuracy** | -4.00% ± 39.80% | Общая точность распознавания |
| **Образцов** | 100 изображений | Количество обработанных изображений |
| **Время** | ~1.3 сек/изображение | Скорость обработки на CPU |

### Интерпретация результатов:

- **Высокие значения WER/CER** указывают на то, что модель требует настройки для конкретной задачи
- **Это нормально** для OCR систем без fine-tuning на специфических данных
- **CER лучше WER** - это типично для OCR систем

## 🔧 Настройка модели

### Смена модели TrOCR

В файле `trocr_evaluation.py` измените строку:

```python
# Текущая модель
evaluator = TrOCREvaluator(model_name="microsoft/trocr-base-handwritten")

# Другие доступные модели:
evaluator = TrOCREvaluator(model_name="microsoft/trocr-large-handwritten")  # Больше, точнее
evaluator = TrOCREvaluator(model_name="microsoft/trocr-base-printed")        # Для печатного текста
evaluator = TrOCREvaluator(model_name="microsoft/trocr-large-printed")       # Большая, печатная
```

### Настройка параметров

В файле `scripts/run_evaluation_real.py` можно изменить:
- Количество обрабатываемых изображений
- Пути к файлам
- Параметры сохранения результатов

## 📈 Анализ результатов

### JSON файл (`trocr_real_iam_evaluation.json`)
Содержит:
- Общую статистику по всем метрикам
- Детальные результаты для каждого изображения
- Информацию о модели и настройках

### CSV файл (`trocr_real_iam_evaluation.csv`)
Содержит таблицу с колонками:
- `ground_truth` - эталонный текст
- `prediction` - распознанный текст
- `wer` - Word Error Rate для данного образца
- `cer` - Character Error Rate для данного образца
- `accuracy` - точность для данного образца

### Графики (`evaluation_real_plots.png`)
Включают:
- Распределение WER
- Распределение CER
- Корреляцию между WER и CER
- Box plot для сравнения метрик

## 🔍 Примеры использования

### Программное использование

```python
from trocr_evaluation import TrOCREvaluator

# Инициализация оценщика
evaluator = TrOCREvaluator()

# Оценка на конкретном изображении
prediction = evaluator.predict_text("path/to/image.png")

# Вычисление метрик
wer = evaluator.calculate_wer("эталонный текст", prediction)
cer = evaluator.calculate_cer("эталонный текст", prediction)

print(f"WER: {wer:.2f}%, CER: {cer:.2f}%")
```

### Анализ результатов в Python

```python
import pandas as pd
import json

# Загрузка результатов
df = pd.read_csv("results/trocr_real_iam_evaluation.csv")

# Статистика
print(f"Средний WER: {df['wer'].mean():.2f}%")
print(f"Средний CER: {df['cer'].mean():.2f}%")

# Лучшие результаты
best_results = df.nsmallest(5, 'wer')
print("Лучшие результаты:")
print(best_results[['ground_truth', 'prediction', 'wer']])
```

## 🛠️ Требования

### Системные требования:
- **Python**: 3.7+
- **RAM**: минимум 8GB (рекомендуется 16GB)
- **Место на диске**: ~5GB (для датасета и модели)
- **GPU**: не обязательно, но значительно ускорит работу

### Установленные зависимости:
- `torch` - PyTorch для работы с моделью
- `transformers` - Hugging Face Transformers
- `jiwer` - библиотека для вычисления WER/CER
- `kagglehub` - для загрузки датасета
- `matplotlib`, `pandas`, `numpy` - для анализа и визуализации

## 🚨 Устранение проблем

### Ошибка "модуль не найден"
```bash
# Активируйте виртуальное окружение
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Переустановите зависимости
pip install -r requirements.txt
```

### Недостаточно памяти
- Закройте другие программы
- Используйте меньшую модель (`trocr-base` вместо `trocr-large`)
- Обрабатывайте меньше изображений

### Медленная работа
- Используйте GPU (автоматически определяется)
- Уменьшите количество обрабатываемых изображений
- Используйте более быструю модель

## 📚 Дополнительная документация

### 🖥️ GUI приложение
- `docs/GUI_README.md` - полная документация GUI
- `docs/UPDATED_BUTTONS_GUIDE.md` - руководство по кнопкам
- `docs/RESULTS_FOLDER_SYSTEM.md` - система папок результатов

### 🔧 Техническая документация
- `docs/PROJECT_GUIDE.md` - руководство по проекту
- `docs/QUICK_START.md` - быстрый старт
- `docs/USAGE_EXAMPLES.md` - примеры использования
- `docs/METRICS_IMPROVEMENTS.md` - улучшения метрик

### 📊 Отчеты и результаты
- `docs/REAL_DATASET_REPORT.md` - анализ результатов
- `docs/FINAL_SUMMARY.md` - итоговый отчет
- `docs/FINAL_*.md` - финальные сводки по функциям

## 🤝 Поддержка

При возникновении проблем:
1. Проверьте активацию виртуального окружения
2. Убедитесь в наличии всех зависимостей
3. Проверьте свободное место на диске
4. Обратитесь к документации в папке `docs/`

## 📄 Лицензия

Проект использует:
- TrOCR модели от Microsoft (MIT License)
- IAM датасет (для исследовательских целей)
- Библиотеки с открытым исходным кодом

---
*Проект создан для исследования и оценки OCR моделей на рукописном тексте*
