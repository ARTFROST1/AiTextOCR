# Примеры использования TrOCR Evaluation

## 🎯 Базовое использование

### Запуск полной оценки
```bash
# 1. Настройка (один раз)
python scripts/setup_environment.py

# 2. Загрузка датасета (один раз)
python scripts/download_real_iam.py

# 3. Оценка модели
python scripts/run_evaluation_real.py
```

## 🔧 Программное использование

### Оценка одного изображения
```python
from trocr_evaluation import TrOCREvaluator

# Инициализация
evaluator = TrOCREvaluator()

# Распознавание текста
prediction = evaluator.predict_text("path/to/image.png")
print(f"Распознанный текст: {prediction}")
```

### Вычисление метрик
```python
# Сравнение с эталоном
ground_truth = "эталонный текст"
prediction = "распознанный текст"

wer = evaluator.calculate_wer(ground_truth, prediction)
cer = evaluator.calculate_cer(ground_truth, prediction)
accuracy = evaluator.calculate_accuracy(ground_truth, prediction)

print(f"WER: {wer:.2f}%")
print(f"CER: {cer:.2f}%")
print(f"Accuracy: {accuracy:.2f}%")
```

### Оценка собственного датасета
```python
# Создание аннотаций
annotations = {
    "image1.png": "текст на первом изображении",
    "image2.png": "текст на втором изображении"
}

# Сохранение аннотаций
import json
with open("my_annotations.json", "w", encoding="utf-8") as f:
    json.dump(annotations, f, ensure_ascii=False, indent=2)

# Оценка
results = evaluator.evaluate_dataset("my_images/", "my_annotations.json")
stats = evaluator.analyze_results(results)

print(f"Средний WER: {stats['mean_wer']:.2f}%")
```

## 📊 Анализ результатов

### Загрузка и анализ CSV
```python
import pandas as pd

# Загрузка результатов
df = pd.read_csv("results/trocr_real_iam_evaluation.csv")

# Базовая статистика
print("Общая статистика:")
print(df[['wer', 'cer', 'accuracy']].describe())

# Лучшие результаты
best_wer = df.nsmallest(5, 'wer')
print("\nЛучшие результаты по WER:")
print(best_wer[['ground_truth', 'prediction', 'wer']])

# Худшие результаты
worst_wer = df.nlargest(5, 'wer')
print("\nХудшие результаты по WER:")
print(worst_wer[['ground_truth', 'prediction', 'wer']])
```

### Анализ JSON результатов
```python
import json

# Загрузка результатов
with open("results/trocr_real_iam_evaluation.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Общая статистика
stats = data['statistics']
print(f"Модель: {data['model_name']}")
print(f"Образцов: {stats['total_samples']}")
print(f"Средний WER: {stats['mean_wer']:.2f}%")

# Анализ детальных результатов
detailed = data['detailed_results']
perfect_matches = [r for r in detailed if r['wer'] == 0.0]
print(f"Идеальных совпадений: {len(perfect_matches)}")
```

## 🎨 Визуализация результатов

### Создание собственных графиков
```python
import matplotlib.pyplot as plt
import pandas as pd

# Загрузка данных
df = pd.read_csv("results/trocr_real_iam_evaluation.csv")

# График распределения WER
plt.figure(figsize=(10, 6))
plt.hist(df['wer'], bins=20, alpha=0.7, color='blue')
plt.title('Распределение Word Error Rate')
plt.xlabel('WER (%)')
plt.ylabel('Частота')
plt.savefig('custom_wer_distribution.png')
plt.show()

# Сравнение WER и CER
plt.figure(figsize=(8, 6))
plt.scatter(df['wer'], df['cer'], alpha=0.6)
plt.title('WER vs CER')
plt.xlabel('Word Error Rate (%)')
plt.ylabel('Character Error Rate (%)')
plt.savefig('custom_wer_vs_cer.png')
plt.show()
```

## 🔄 Сравнение моделей

### Тестирование разных моделей TrOCR
```python
models = [
    "microsoft/trocr-base-handwritten",
    "microsoft/trocr-large-handwritten",
    "microsoft/trocr-base-printed"
]

results_comparison = {}

for model_name in models:
    print(f"\nТестирование модели: {model_name}")
    
    try:
        evaluator = TrOCREvaluator(model_name=model_name)
        
        # Оценка на подмножестве данных
        results = evaluator.evaluate_dataset("iam_dataset/images", "iam_dataset/kaggle_annotations.json")
        stats = evaluator.analyze_results(results)
        
        results_comparison[model_name] = {
            'wer': stats['mean_wer'],
            'cer': stats['mean_cer'],
            'accuracy': stats['mean_accuracy']
        }
        
    except Exception as e:
        print(f"Ошибка с моделью {model_name}: {e}")

# Вывод сравнения
print("\nСравнение моделей:")
for model, metrics in results_comparison.items():
    print(f"{model}:")
    print(f"  WER: {metrics['wer']:.2f}%")
    print(f"  CER: {metrics['cer']:.2f}%")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
```

## 🛠️ Настройка параметров

### Изменение количества обрабатываемых изображений
В файле `scripts/download_real_iam.py` измените:
```python
# Строка 206: ограничение количества изображений
for i, image_file in enumerate(image_files[:50]):  # Было 100, стало 50
```

### Изменение параметров генерации модели
В файле `trocr_evaluation.py` добавьте параметры:
```python
def predict_text(self, image_path: str) -> str:
    # ... существующий код ...
    
    with torch.no_grad():
        generated_ids = self.model.generate(
            pixel_values,
            max_length=512,        # Максимальная длина текста
            num_beams=5,           # Количество лучей для beam search
            early_stopping=True    # Остановка при завершении
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()
```

## 📁 Работа с собственными данными

### Подготовка собственного датасета
```python
import os
import json
from PIL import Image

def prepare_custom_dataset(image_folder, output_folder):
    """Подготовка собственного датасета"""
    
    # Создание аннотаций
    annotations = {}
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            
            # Здесь можно добавить логику извлечения текста из имени файла
            # или использовать другой способ получения эталонного текста
            text = filename.split('.')[0].replace('_', ' ')
            
            annotations[image_path] = text
    
    # Сохранение аннотаций
    os.makedirs(output_folder, exist_ok=True)
    annotations_file = os.path.join(output_folder, "custom_annotations.json")
    
    with open(annotations_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    
    print(f"Аннотации сохранены в {annotations_file}")
    return annotations_file

# Использование
annotations_file = prepare_custom_dataset("my_images/", "custom_dataset/")

# Оценка на собственных данных
evaluator = TrOCREvaluator()
results = evaluator.evaluate_dataset("my_images/", annotations_file)
```

## 🔍 Отладка и диагностика

### Проверка качества изображений
```python
from PIL import Image
import os

def analyze_image_quality(image_folder):
    """Анализ качества изображений"""
    
    issues = []
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    
                    if width < 50 or height < 50:
                        issues.append(f"{filename}: слишком маленькое изображение ({width}x{height})")
                    
                    if img.mode != 'RGB':
                        issues.append(f"{filename}: неправильный режим цвета ({img.mode})")
                        
            except Exception as e:
                issues.append(f"{filename}: ошибка загрузки ({e})")
    
    return issues

# Проверка качества
issues = analyze_image_quality("iam_dataset/images/")
if issues:
    print("Найдены проблемы с изображениями:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Все изображения в порядке!")
```

## 📚 Дополнительные ресурсы

- [Hugging Face TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten)
- [IAM Dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- [JIWER Documentation](https://github.com/jitsi/jiwer)

---
*Эти примеры помогут вам максимально эффективно использовать систему оценки TrOCR*
