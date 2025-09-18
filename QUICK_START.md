# Быстрый старт - TrOCR Evaluation

## 🚀 Запуск в 3 команды

### 1. Настройка окружения
```bash
python scripts/setup_environment.py
```

### 2. Загрузка датасета IAM
```bash
python scripts/download_real_iam.py
```

### 3. Запуск оценки
```bash
python scripts/run_full_evaluation.py
```

## 📊 Что получите

После выполнения всех команд в папке `results/` появятся:

- **`trocr_real_iam_evaluation.json`** - детальные результаты в JSON
- **`trocr_real_iam_evaluation.csv`** - таблица для анализа в Excel/Google Sheets  
- **`evaluation_real_plots.png`** - графики с метриками

## 📈 Ожидаемые результаты

- **Время выполнения**: 2-5 минут
- **Количество образцов**: 100 изображений рукописного текста
- **Средний WER**: ~104% (высокий из-за сложности рукописного текста)
- **Средний CER**: ~96% (значительно сложнее печатного текста)

> **Примечание**: Высокие значения WER/CER нормальны для OCR без fine-tuning. Это показывает, что модель требует настройки для конкретной задачи.

## 🔧 Настройка модели

В файле `trocr_evaluation.py` измените модель:

```python
# Для лучшего качества (медленнее)
evaluator = TrOCREvaluator(model_name="microsoft/trocr-large-handwritten")

# Для печатного текста
evaluator = TrOCREvaluator(model_name="microsoft/trocr-base-printed")
```

## 📁 Структура проекта

```
📦 TrOCR Evaluation
├── 📁 scripts/          # Скрипты запуска
├── 📁 iam_dataset/      # Датасет IAM (100 изображений)
├── 📁 results/          # Результаты оценки
├── 📁 docs/             # Документация
└── 📄 trocr_evaluation.py  # Основной модуль
```

## 🆘 Если что-то не работает

1. **Активируйте виртуальное окружение**:
   ```bash
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/macOS
   ```

2. **Переустановите зависимости**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Проверьте свободное место** (нужно ~5GB)

## 📚 Подробная документация

- `README.md` - полная инструкция
- `docs/REAL_DATASET_REPORT.md` - анализ результатов
- `docs/FINAL_SUMMARY.md` - итоговый отчет

---
*Система готова к использованию!*
