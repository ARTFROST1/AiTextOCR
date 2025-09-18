# Структура проекта TrOCR Evaluation

## 📁 Организация файлов

```
📦 TrOCR Evaluation Project
├── 📄 README.md                    # Основная документация
├── 📄 QUICK_START.md               # Быстрый старт (3 команды)
├── 📄 PROJECT_GUIDE.md             # Руководство по проекту
├── 📄 PROJECT_STRUCTURE.md         # Эта структура (этот файл)
├── 📄 USAGE_EXAMPLES.md            # Примеры использования
├── 📄 requirements.txt             # Зависимости Python
├── 📄 trocr_evaluation.py          # Основной модуль оценки
│
├── 📁 scripts/                     # Скрипты для работы
│   ├── setup_environment.py       # Настройка окружения
│   ├── download_real_iam.py       # Загрузка датасета IAM
│   └── run_full_evaluation.py     # Полный запуск оценки модели
│
├── 📁 iam_dataset/                # Датасет IAM
│   ├── images/                    # 100 изображений рукописного текста
│   │   ├── image_0000.png
│   │   ├── image_0001.png
│   │   └── ... (до image_0099.png)
│   └── kaggle_annotations.json    # Аннотации к изображениям
│
├── 📁 results/                    # Результаты оценки
│   ├── trocr_real_iam_evaluation.json  # Детальные результаты (JSON)
│   ├── trocr_real_iam_evaluation.csv   # Результаты в табличном формате
│   └── evaluation_real_plots.png       # Графики метрик
│
├── 📁 docs/                       # Документация и отчеты
│   ├── REAL_DATASET_REPORT.md     # Отчет по результатам на реальных данных
│   └── FINAL_SUMMARY.md           # Итоговый отчет по проекту
│
├── 📄 .gitignore                  # Исключения для Git
└── 📁 venv/                       # Виртуальное окружение Python
    ├── Scripts/                   # Исполняемые файлы
    ├── Lib/                       # Библиотеки Python
    └── pyvenv.cfg                 # Конфигурация окружения
```

## 🎯 Назначение каждого компонента

### 📄 Основные файлы

| Файл | Назначение | Когда использовать |
|------|------------|-------------------|
| `README.md` | Полная техническая документация | Для изучения всех возможностей |
| `QUICK_START.md` | Быстрый старт в 3 команды | Для немедленного запуска |
| `PROJECT_GUIDE.md` | Руководство по проекту | Для понимания архитектуры |
| `USAGE_EXAMPLES.md` | Примеры программного использования | Для разработки и экспериментов |

### 📁 Папка scripts/

| Скрипт | Функция | Запуск |
|--------|---------|--------|
| `setup_environment.py` | Настройка окружения, установка зависимостей | `python scripts/setup_environment.py` |
| `download_real_iam.py` | Загрузка датасета IAM через KaggleHub | `python scripts/download_real_iam.py` |
| `run_full_evaluation.py` | Полный запуск оценки модели | `python scripts/run_full_evaluation.py` |

### 📁 Папка iam_dataset/

| Компонент | Содержимое | Размер |
|-----------|------------|--------|
| `images/` | 100 PNG изображений рукописного текста | ~50MB |
| `kaggle_annotations.json` | Аннотации с эталонным текстом | ~10KB |

### 📁 Папка results/

| Файл | Формат | Назначение |
|------|--------|------------|
| `trocr_real_iam_evaluation.json` | JSON | Детальные результаты, программный анализ |
| `trocr_real_iam_evaluation.csv` | CSV | Таблица для Excel, статистический анализ |
| `evaluation_real_plots.png` | PNG | Графики метрик, презентации |

### 📁 Папка docs/

| Файл | Содержимое | Аудитория |
|------|------------|-----------|
| `REAL_DATASET_REPORT.md` | Анализ результатов на реальных данных | Исследователи, аналитики |
| `FINAL_SUMMARY.md` | Итоговый отчет по проекту | Руководители, заказчики |

## 🔄 Жизненный цикл проекта

### 1. Инициализация
```bash
python scripts/setup_environment.py
```
**Результат**: Создается виртуальное окружение, устанавливаются зависимости

### 2. Подготовка данных
```bash
python scripts/download_real_iam.py
```
**Результат**: Скачивается датасет IAM, создается структура `iam_dataset/`

### 3. Выполнение оценки
```bash
python scripts/run_evaluation_real.py
```
**Результат**: Создается папка `results/` с результатами оценки

### 4. Анализ результатов
- Открыть `results/trocr_real_iam_evaluation.csv` в Excel
- Изучить `results/evaluation_real_plots.png`
- Прочитать `docs/REAL_DATASET_REPORT.md`

## 📊 Типы файлов и их использование

### Данные
- **`.png`** - изображения рукописного текста (входные данные)
- **`.json`** - аннотации и результаты (структурированные данные)

### Код
- **`.py`** - Python скрипты и модули
- **`.txt`** - конфигурационные файлы (requirements.txt)

### Документация
- **`.md`** - Markdown документация
- **`.png`** - графики и визуализации

### Результаты
- **`.csv`** - табличные данные для анализа
- **`.json`** - детальные результаты для программной обработки

## 🎨 Принципы организации

### 1. Разделение по типу
- **Код** → `scripts/`, корень проекта
- **Данные** → `iam_dataset/`
- **Результаты** → `results/`
- **Документация** → `docs/`, корень проекта

### 2. Логическая группировка
- Все скрипты запуска в `scripts/`
- Вся документация в `docs/`
- Все результаты в `results/`

### 3. Четкие имена
- `setup_environment.py` - настройка
- `download_real_iam.py` - загрузка
- `run_evaluation_real.py` - запуск
- `trocr_real_iam_evaluation.json` - результаты

### 4. Исключение временных файлов
- Удалены тестовые данные
- Убраны временные результаты
- Оставлены только настоящие данные

## 🚀 Быстрая навигация

### Для запуска проекта:
1. `QUICK_START.md` → 3 команды
2. `scripts/setup_environment.py` → настройка
3. `scripts/download_real_iam.py` → данные
4. `scripts/run_evaluation_real.py` → оценка

### Для изучения результатов:
1. `results/trocr_real_iam_evaluation.csv` → таблица
2. `results/evaluation_real_plots.png` → графики
3. `docs/REAL_DATASET_REPORT.md` → анализ

### Для разработки:
1. `USAGE_EXAMPLES.md` → примеры кода
2. `trocr_evaluation.py` → основной модуль
3. `PROJECT_GUIDE.md` → архитектура

### Для документации:
1. `README.md` → полное описание
2. `PROJECT_GUIDE.md` → руководство
3. `docs/FINAL_SUMMARY.md` → итоги

## 🔧 Настройка структуры

### Добавление новых скриптов:
```bash
# Поместить в scripts/
mv new_script.py scripts/
```

### Добавление новых результатов:
```bash
# Поместить в results/
mv new_results.json results/
```

### Добавление документации:
```bash
# Поместить в docs/ или корень
mv new_docs.md docs/
```

### Изменение путей в скриптах:
Все скрипты настроены на работу с новой структурой:
- Результаты сохраняются в `results/`
- Данные берутся из `iam_dataset/`
- Документация в `docs/`

---
*Структура проекта оптимизирована для удобства использования и поддержки*
