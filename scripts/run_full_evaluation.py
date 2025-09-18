"""
Полный скрипт для запуска всей оценки от начала до конца
"""

import os
import sys
import subprocess
from pathlib import Path
import json
import time

def get_python_executable():
    """Возвращает путь к интерпретатору из venv, если он существует, иначе текущий."""
    # приоритет CUDA venv, затем обычный venv, затем системный
    venv_cuda_python = Path("venv_cuda") / "Scripts" / "python.exe"
    if venv_cuda_python.exists():
        return str(venv_cuda_python)
    venv_python = Path("venv") / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable

def run_script(script_name, description):
    """Запуск скрипта с выводом описания"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    try:
        python_exec = get_python_executable()
        result = subprocess.run([python_exec, script_name], check=True)
        print(f"✓ {description} завершено успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Ошибка при выполнении {script_name}: {e}")
        return False
    except FileNotFoundError:
        print(f"✗ Файл {script_name} не найден")
        return False

def check_environment():
    """Проверка готовности окружения"""
    print("Проверка готовности окружения...")
    
    # Проверяем наличие основных файлов
    required_files = [
        "trocr_evaluation.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ Отсутствуют файлы: {missing_files}")
        return False
    
    # Проверяем виртуальное окружение (поддержка venv_cuda и venv)
    if not (Path("venv_cuda").exists() or Path("venv").exists()):
        print("✗ Виртуальное окружение не найдено")
        print("Запустите: python setup_environment.py")
        return False
    
    print("✓ Окружение готово")
    return True

def check_dataset():
    """Проверка наличия датасета"""
    print("\nПроверка датасета...")
    
    dataset_paths = [
        "IAM/image"
    ]
    
    annotation_paths = [
        "IAM/gt_test.txt"
    ]
    
    # Ищем существующий датасет
    dataset_found = False
    annotation_found = False
    
    for dataset_path in dataset_paths:
        if Path(dataset_path).exists():
            print(f"✓ Найден датасет: {dataset_path}")
            dataset_found = True
            break
    
    for annotation_path in annotation_paths:
        if Path(annotation_path).exists():
            print(f"✓ Найдены аннотации: {annotation_path}")
            annotation_found = True
            break
    
    if not dataset_found or not annotation_found:
        print("ℹ Датасет не найден, будет создан тестовый датасет")
        return False
    
    return True

def run_full_evaluation():
    """Запуск полной оценки"""
    print("\n" + "="*60)
    print("ЗАПУСК ПОЛНОЙ ОЦЕНКИ TrOCR НА ДАТАСЕТЕ IAM")
    print("="*60)
    
    start_time = time.time()
    
    # Проверяем окружение
    if not check_environment():
        print("\nСначала настройте окружение:")
        print("python setup_environment.py")
        return False
    
    # Проверяем/создаем датасет
    if not check_dataset():
        print("\nДатасет не найден. Поместите изображения в 'IAM/image' и файл аннотаций в 'IAM/gt_test.txt'.")
        return False
    
    # Запускаем оценку напрямую
    print("\n" + "="*60)
    print("🚀 ЗАПУСК ОЦЕНКИ МОДЕЛИ TrOCR")
    print("="*60)
    
    try:
        # Добавляем путь к корневой директории проекта
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        from trocr_evaluation import TrOCREvaluator
        import json
        from pathlib import Path
        
        # Ищем файлы датасета
        dataset_path = "IAM/image"
        annotations_path = "IAM/gt_test.txt"
        
        if not (Path(dataset_path).exists() and Path(annotations_path).exists()):
            print("❌ Датасет не найден!")
            return False
        
        # Подсчитываем количество изображений
        image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(image_files) == 0:
            print("❌ Изображения не найдены в директории!")
            return False
        
        print(f"📊 Обрабатывается: {min(50, len(image_files))} из {len(image_files)} изображений")
        print("⏱️  Инициализация модели...")
        
        # Инициализация оценщика
        evaluator = TrOCREvaluator()
        
        print("🔄 Начинается обработка изображений...")
        
        # Оценка на датасете (первые 50)
        results = evaluator.evaluate_dataset(dataset_path, annotations_path, limit=50)
        
        # Анализ результатов
        stats = evaluator.analyze_results(results)
        
        # Сохранение результатов
        output_file = "results/trocr_real_iam_evaluation.json"
        evaluator.save_results(results, stats, output_file)
        
        # Создание графиков
        evaluator.plot_results(results, "results/evaluation_real_plots.png", show=False)
        
        print("\n" + "="*60)
        print("📊 РЕЗУЛЬТАТЫ ОЦЕНКИ")
        print("="*60)
        
        # Красивое отображение статистики
        print(f"📈 Обработано образцов: {stats['total_samples']}")
        print("\n")
        print(f"📉 Средний WER:        {stats['mean_wer']:.2f}%")
        print(f"📈 Средний CER:        {stats['mean_cer']:.2f}%")
        print(f"📊 Средняя точность:   {stats['mean_accuracy']:.2f}%")
        print("\n")
        print(f"📉 Стандартное отклонение WER: {stats['std_wer']:.2f}%")
        print(f"📈 Стандартное отклонение CER: {stats['std_cer']:.2f}%")
        
        # Оценка качества
        print(f"\n🎯 ОЦЕНКА КАЧЕСТВА:")
        if stats['mean_wer'] < 20:
            print("🟢 Отличное качество распознавания!")
        elif stats['mean_wer'] < 40:
            print("🟡 Хорошее качество распознавания")
        elif stats['mean_wer'] < 60:
            print("🟠 Удовлетворительное качество")
        else:
            print("🔴 Низкое качество распознавания")
        
        print(f"📈 CER в {stats['mean_cer'] / stats['mean_wer']:.1f} раз лучше WER")
        
        # Информация о файлах
        print(f"\n💾 СОЗДАННЫЕ ФАЙЛЫ:")
        print(f"📄 JSON: {output_file}")
        print(f"📊 CSV:  {output_file.replace('.json', '.csv')}")
        print(f"📈 PNG:  results/evaluation_real_plots.png")
        
    except Exception as e:
        print(f"❌ Ошибка при оценке: {e}")
        return False
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("✅ ОЦЕНКА ЗАВЕРШЕНА УСПЕШНО!")
    print("="*60)
    print(f"⏱️  Время выполнения: {duration:.1f} секунд")
    print(f"📁 Результаты сохранены в папке 'results/'")
    
    return True

def main():
    """Основная функция"""
    try:
        success = run_full_evaluation()
        
        if success:
            print("\n🎉 Проект готов к использованию!")
        else:
            print("\n❌ Произошли ошибки. Проверьте вывод выше.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nОстановлено пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\nНеожиданная ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
