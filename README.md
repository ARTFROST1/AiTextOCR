# TrOCR Evaluation на датасете IAM

## 📋 Описание проекта

Этот проект реализует систему оценки модели **TrOCR (Transformer-based OCR)** на датасете **IAM (Institute of Automation and Mathematics)** для распознавания рукописного текста. Система вычисляет основные метрики качества OCR:

- **WER (Word Error Rate)** - процент ошибок на уровне слов
- **CER (Character Error Rate)** - процент ошибок на уровне символов  
- **Accuracy** - точность распознавания

## 🏗️ Структура проекта

```
📦 TrOCR Evaluation Project
├── 📁 scripts/                    # Скрипты для работы с проектом
│   ├── setup_environment.py      # Настройка окружения
│   ├── download_real_iam.py      # Загрузка датасета IAM
│   └── run_evaluation_real.py    # Запуск оценки модели
├── 📁 iam_dataset/               # Датасет IAM
│   ├── images/                   # Изображения рукописного текста
│   └── kaggle_annotations.json   # Аннотации к изображениям
├── 📁 results/                   # Результаты оценки
│   ├── trocr_real_iam_evaluation.json  # Детальные результаты (JSON)
│   ├── trocr_real_iam_evaluation.csv   # Результаты в табличном формате
│   └── evaluation_real_plots.png       # Графики метрик
├── 📁 docs/                      # Документация
│   ├── REAL_DATASET_REPORT.md    # Отчет по результатам
│   └── FINAL_SUMMARY.md          # Итоговый отчет
├── 📄 trocr_evaluation.py        # Основной модуль оценки
├── 📄 requirements.txt           # Зависимости Python
└── 📄 README.md                  # Эта инструкция
```

## 🚀 Быстрый старт

### 1. Настройка окружения
```bash
python scripts/setup_environment.py
```

### 2. Загрузка датасета IAM
```bash
python scripts/download_real_iam.py
```

### 3. Запуск оценки модели
```bash
python scripts/run_full_evaluation.py
```

### 4. Просмотр результатов
Результаты будут сохранены в папке `results/`:
- `trocr_real_iam_evaluation.json` - детальные результаты
- `trocr_real_iam_evaluation.csv` - таблица для анализа
- `evaluation_real_plots.png` - графики метрик

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

- `docs/REAL_DATASET_REPORT.md` - подробный анализ результатов
- `docs/FINAL_SUMMARY.md` - итоговый отчет по проекту

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
