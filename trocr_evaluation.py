"""
TrOCR Evaluation on IAM Dataset
Оценка модели TrOCR на датасете IAM с вычислением метрик WER, CER и Accuracy
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# TrOCR и метрики
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import jiwer

class TrOCREvaluator:
    """Класс для оценки модели TrOCR на датасете IAM"""
    
    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten"):
        """
        Инициализация оценщика
        
        Args:
            model_name: Название модели TrOCR
        """
        self.model_name = model_name
        # Предпочитаем GPU, если доступен
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device.type == "cuda":
            try:
                device_name = torch.cuda.get_device_name(0)
            except Exception:
                device_name = "CUDA"
            print(f"Используется устройство: cuda ({device_name}), gpus={torch.cuda.device_count()}, cuda={torch.version.cuda}")
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
        else:
            print(f"Используется устройство: cpu")
        print(f"Загружается модель: {model_name}")
        
        # Загрузка модели и процессора
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("Модель успешно загружена!")
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Предобработка изображения для модели
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Обработанное изображение
        """
        try:
            image = Image.open(image_path).convert('RGB')
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            return pixel_values.to(self.device)
        except Exception as e:
            print(f"Ошибка при обработке изображения {image_path}: {e}")
            return None
    
    def predict_text(self, image_path: str) -> str:
        """
        Распознавание текста на изображении
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Распознанный текст
        """
        pixel_values = self.preprocess_image(image_path)
        if pixel_values is None:
            return ""
        
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return generated_text.strip()
        except Exception as e:
            print(f"Ошибка при распознавании текста {image_path}: {e}")
            return ""
    
    def calculate_metrics(self, reference: str, hypothesis: str):
        """
        Расчет WER и CER по стандарту (Левенштейн) с нормализацией текста
        
        Returns:
            (WER%, CER%)
        """
        # Нормализация: привести к нижнему регистру, убрать лишние пробелы
        norm = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.Strip(),
            jiwer.RemoveMultipleSpaces(),
        ])

        ref = norm(reference if reference is not None else "")
        hyp = norm(hypothesis if hypothesis is not None else "")

        # edge-cases: пустая ссылка и/или предсказание
        if ref == "" and hyp == "":
            return 0.0, 0.0
        if ref == "" and hyp != "":
            return 100.0, 100.0

        try:
            wer = jiwer.wer(ref, hyp) * 100.0
        except Exception:
            wer = 100.0

        try:
            cer = jiwer.cer(ref, hyp) * 100.0
        except Exception:
            cer = 100.0

        return wer, cer

    def calculate_wer(self, ground_truth: str, prediction: str) -> float:
        """
        Вычисление Word Error Rate (WER)
        
        Args:
            ground_truth: Эталонный текст
            prediction: Предсказанный текст
            
        Returns:
            WER в процентах
        """
        wer, _ = self.calculate_metrics(ground_truth, prediction)
        return wer
    
    def calculate_cer(self, ground_truth: str, prediction: str) -> float:
        """
        Вычисление Character Error Rate (CER)
        
        Args:
            ground_truth: Эталонный текст
            prediction: Предсказанный текст
            
        Returns:
            CER в процентах
        """
        _, cer = self.calculate_metrics(ground_truth, prediction)
        return cer
    
    def calculate_accuracy(self, ground_truth: str, prediction: str) -> float:
        """
        Вычисление точности (Accuracy)
        
        Args:
            ground_truth: Эталонный текст
            prediction: Предсказанный текст
            
        Returns:
            Accuracy в процентах
        """
        # Определим accuracy на уровне символов как 100 * (1 - CER)
        cer = self.calculate_cer(ground_truth, prediction)
        return max(0.0, 100.0 - cer)
    
    def evaluate_dataset(self, dataset_path: str, annotations_file: str = None, limit: int = 50) -> Dict:
        """
        Оценка модели на датасете
        
        Args:
            dataset_path: Путь к папке с изображениями
            annotations_file: Путь к файлу с аннотациями
            
        Returns:
            Словарь с результатами оценки
        """
        results = {
            'predictions': [],
            'ground_truths': [],
            'wer_scores': [],
            'cer_scores': [],
            'accuracy_scores': []
        }
        
        # Если аннотации не предоставлены, используем имена файлов
        if annotations_file and os.path.exists(annotations_file):
            annotations = self.load_annotations(annotations_file, dataset_path)
        else:
            annotations = self.create_annotations_from_filenames(dataset_path)
        
        # Ограничиваемся первыми N примерами для ускорения
        if limit is not None and limit > 0:
            annotations = dict(list(annotations.items())[:limit])
        
        print(f"Начинается оценка на {len(annotations)} изображениях...")
        
        for image_path, ground_truth in tqdm(annotations.items(), desc="Обработка изображений"):
            if not os.path.exists(image_path):
                print(f"Файл не найден: {image_path}")
                continue
            
            # Распознавание текста
            prediction = self.predict_text(image_path)
            
            # Вычисление метрик
            wer = self.calculate_wer(ground_truth, prediction)
            cer = self.calculate_cer(ground_truth, prediction)
            accuracy = self.calculate_accuracy(ground_truth, prediction)
            
            # Сохранение результатов
            results['predictions'].append(prediction)
            results['ground_truths'].append(ground_truth)
            results['wer_scores'].append(wer)
            results['cer_scores'].append(cer)
            results['accuracy_scores'].append(accuracy)
        
        return results
    
    def load_annotations(self, annotations_file: str, images_dir: str) -> Dict[str, str]:
        """
        Загрузка аннотаций из файла
        
        Args:
            annotations_file: Путь к файлу с аннотациями
            
        Returns:
            Словарь {путь_к_изображению: текст}
        """
        annotations = {}
        
        if annotations_file.endswith('.json'):
            with open(annotations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                annotations = data
        elif annotations_file.endswith('.txt'):
            with open(annotations_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        image_name, text = parts
                        # Собираем абсолютный путь к изображению из директории и имени файла
                        image_path = os.path.join(images_dir, image_name)
                        annotations[image_path] = text
        
        return annotations
    
    def create_annotations_from_filenames(self, dataset_path: str) -> Dict[str, str]:
        """
        Создание аннотаций на основе имен файлов (для демонстрации)
        
        Args:
            dataset_path: Путь к папке с изображениями
            
        Returns:
            Словарь {путь_к_изображению: текст}
        """
        annotations = {}
        
        for filename in os.listdir(dataset_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(dataset_path, filename)
                # Простое извлечение текста из имени файла (нужно адаптировать под ваш датасет)
                text = filename.split('.')[0].replace('_', ' ')
                annotations[image_path] = text
        
        return annotations
    
    def analyze_results(self, results: Dict) -> Dict:
        """
        Анализ результатов оценки
        
        Args:
            results: Результаты оценки
            
        Returns:
            Статистика результатов
        """
        stats = {
            'total_samples': len(results['wer_scores']),
            'mean_wer': np.mean(results['wer_scores']),
            'std_wer': np.std(results['wer_scores']),
            'mean_cer': np.mean(results['cer_scores']),
            'std_cer': np.std(results['cer_scores']),
            'mean_accuracy': np.mean(results['accuracy_scores']),
            'std_accuracy': np.std(results['accuracy_scores']),
            'min_wer': np.min(results['wer_scores']),
            'max_wer': np.max(results['wer_scores']),
            'min_cer': np.min(results['cer_scores']),
            'max_cer': np.max(results['cer_scores'])
        }
        
        return stats
    
    def plot_results(self, results: Dict, save_path: str = None, show: bool = False):
        """
        Визуализация результатов
        
        Args:
            results: Результаты оценки
            save_path: Путь для сохранения графиков
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Распределение WER
        axes[0, 0].hist(results['wer_scores'], bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_title('Распределение WER')
        axes[0, 0].set_xlabel('WER (%)')
        axes[0, 0].set_ylabel('Частота')
        
        # Распределение CER
        axes[0, 1].hist(results['cer_scores'], bins=30, alpha=0.7, color='green')
        axes[0, 1].set_title('Распределение CER')
        axes[0, 1].set_xlabel('CER (%)')
        axes[0, 1].set_ylabel('Частота')
        
        # Сравнение WER и CER
        axes[1, 0].scatter(results['wer_scores'], results['cer_scores'], alpha=0.6)
        axes[1, 0].set_title('WER vs CER')
        axes[1, 0].set_xlabel('WER (%)')
        axes[1, 0].set_ylabel('CER (%)')
        
        # Box plot метрик
        data_for_box = [results['wer_scores'], results['cer_scores']]
        axes[1, 1].boxplot(data_for_box, labels=['WER', 'CER'])
        axes[1, 1].set_title('Распределение метрик')
        axes[1, 1].set_ylabel('Процент ошибок')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        plt.close(fig)
    
    def save_results(self, results: Dict, stats: Dict, output_path: str = "evaluation_results.json"):
        """
        Сохранение результатов в файл
        
        Args:
            results: Результаты оценки
            stats: Статистика результатов
            output_path: Путь для сохранения
        """
        output_data = {
            'model_name': self.model_name,
            'statistics': stats,
            'detailed_results': []
        }
        
        for i in range(len(results['predictions'])):
            output_data['detailed_results'].append({
                'ground_truth': results['ground_truths'][i],
                'prediction': results['predictions'][i],
                'wer': results['wer_scores'][i],
                'cer': results['cer_scores'][i],
                'accuracy': results['accuracy_scores'][i]
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # Также сохраняем CSV для удобного анализа
        df = pd.DataFrame({
            'ground_truth': results['ground_truths'],
            'prediction': results['predictions'],
            'wer': results['wer_scores'],
            'cer': results['cer_scores'],
            'accuracy': results['accuracy_scores']
        })
        
        csv_path = output_path.replace('.json', '.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')


def main():
    """Основная функция для запуска оценки"""
    
    # Инициализация оценщика
    evaluator = TrOCREvaluator()
    
    # Пути к данным нового датасета IAM
    dataset_path = "IAM/image"  # Путь к изображениям IAM (.jpg)
    annotations_file = "IAM/gt_test.txt"  # Путь к расшифровкам
    
    print("Начинается оценка модели TrOCR на датасете IAM...")
    
    # Оценка модели (первые 50 примеров)
    results = evaluator.evaluate_dataset(dataset_path, annotations_file, limit=50)
    
    # Анализ результатов
    stats = evaluator.analyze_results(results)
    
    # Визуализация результатов
    evaluator.plot_results(results, "results/evaluation_real_plots.png", show=False)
    
    # Сохранение результатов
    evaluator.save_results(results, stats, "results/trocr_real_iam_evaluation.json")


if __name__ == "__main__":
    main()
