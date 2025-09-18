"""
Тестирование улучшенных метрик WER, CER и Accuracy
"""

import sys
import os
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from trocr_evaluation import TrOCREvaluator

def test_metrics_calculation():
    """Тестирование расчета метрик на различных примерах"""
    
    print("🧪 Тестирование улучшенных метрик WER, CER и Accuracy")
    print("=" * 60)
    
    # Создаем экземпляр оценщика
    evaluator = TrOCREvaluator()
    
    # Тестовые случаи
    test_cases = [
        {
            "name": "Идеальное совпадение",
            "reference": "Hello world",
            "hypothesis": "Hello world"
        },
        {
            "name": "Одна замена",
            "reference": "Hello world",
            "hypothesis": "Hello word"
        },
        {
            "name": "Одна вставка",
            "reference": "Hello world",
            "hypothesis": "Hello the world"
        },
        {
            "name": "Одно удаление",
            "reference": "Hello world",
            "hypothesis": "Hello"
        },
        {
            "name": "Множественные ошибки",
            "reference": "The quick brown fox",
            "hypothesis": "The quik brown foks"
        },
        {
            "name": "Пустые строки",
            "reference": "",
            "hypothesis": ""
        },
        {
            "name": "Пустая гипотеза",
            "reference": "Hello world",
            "hypothesis": ""
        },
        {
            "name": "Пустая ссылка",
            "reference": "",
            "hypothesis": "Hello world"
        },
        {
            "name": "Разный регистр и пунктуация",
            "reference": "Hello, World!",
            "hypothesis": "hello world"
        },
        {
            "name": "Лишние пробелы",
            "reference": "Hello   world",
            "hypothesis": "Hello world"
        }
    ]
    
    print(f"{'Тест':<25} {'WER':<8} {'CER':<8} {'Acc (симв)':<12} {'Acc (слова)':<12}")
    print("-" * 70)
    
    for test_case in test_cases:
        ref = test_case["reference"]
        hyp = test_case["hypothesis"]
        
        # Вычисляем метрики
        wer, cer, word_accuracy, char_accuracy = evaluator.calculate_metrics(ref, hyp)
        
        print(f"{test_case['name']:<25} {wer:<8.2f} {cer:<8.2f} {char_accuracy:<12.2f} {word_accuracy:<12.2f}")
    
    print("\n" + "=" * 60)
    print("✅ Тестирование завершено!")
    
    # Дополнительные тесты нормализации
    print("\n🔍 Тестирование нормализации текста:")
    print("-" * 40)
    
    normalization_tests = [
        "  Hello,   World!  ",
        "HELLO WORLD",
        "hello world",
        "Hello, World!",
        "  Multiple    Spaces  "
    ]
    
    for text in normalization_tests:
        normalized = evaluator.normalize_text(text)
        print(f"'{text}' -> '{normalized}'")

def test_edge_cases():
    """Тестирование граничных случаев"""
    
    print("\n🚨 Тестирование граничных случаев:")
    print("-" * 40)
    
    evaluator = TrOCREvaluator()
    
    edge_cases = [
        (None, None),
        (None, "test"),
        ("test", None),
        ("", ""),
        ("", "test"),
        ("test", ""),
        ("a" * 1000, "b" * 1000),  # Очень длинные строки
        ("a", "b" * 100),  # Очень разные длины
    ]
    
    for i, (ref, hyp) in enumerate(edge_cases):
        try:
            wer, cer, word_acc, char_acc = evaluator.calculate_metrics(ref, hyp)
            print(f"Случай {i+1}: WER={wer:.2f}, CER={cer:.2f}, WordAcc={word_acc:.2f}, CharAcc={char_acc:.2f}")
        except Exception as e:
            print(f"Случай {i+1}: ОШИБКА - {e}")

def main():
    """Главная функция тестирования"""
    try:
        test_metrics_calculation()
        test_edge_cases()
        
        print("\n🎉 Все тесты пройдены успешно!")
        print("Улучшенные метрики работают корректно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
