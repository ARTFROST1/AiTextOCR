"""
EasyOCR Evaluation on IAM Dataset
Оценка модели EasyOCR на датасете IAM с вычислением метрик WER, CER и Accuracy
Совместимый интерфейс с TrOCREvaluator для прозрачной подстановки в GUI
"""

import os
import json
from typing import Dict

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import jiwer

try:
    import easyocr
except ImportError as e:
    raise ImportError("easyocr не установлен. Добавьте 'easyocr' в requirements.txt и установите зависимости.")


class EasyOCREvaluator:
    """Класс для оценки EasyOCR с интерфейсом, совместимым с TrOCREvaluator"""

    def __init__(self, langs: str = "en"):
        """
        Args:
            langs: Строка языков, например "en", "ru", "ru+en"
        """
        self.model_name = f"easyocr:{langs}"
        self.langs = [p for p in langs.replace(',', '+').split('+') if p]
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
            print("Используется устройство: cpu")

        print(f"Загружается EasyOCR Reader с языками: {self.langs}")
        # gpu=True автоматически использует доступный CUDA через torch
        self.reader = easyocr.Reader(self.langs, gpu=(self.device.type == "cuda"))
        print("EasyOCR успешно инициализирован!")

    # ----- Метрики и нормализация (совместимы с TrOCREvaluator) -----
    def normalize_text(self, text: str) -> str:
        if text is None:
            return ""
        norm = jiwer.Compose([
            jiwer.Strip(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemoveEmptyStrings(),
        ])
        return norm(text)

    def calculate_metrics(self, reference: str, hypothesis: str):
        ref = self.normalize_text(reference)
        hyp = self.normalize_text(hypothesis)

        if ref == "" and hyp == "":
            return 0.0, 0.0, 100.0, 100.0
        if ref == "" and hyp != "":
            return 100.0, 100.0, 0.0, 0.0
        if ref != "" and hyp == "":
            return 100.0, 100.0, 0.0, 0.0

        try:
            wer = jiwer.wer(ref, hyp) * 100.0
            cer = jiwer.cer(ref, hyp) * 100.0
            word_accuracy = max(0.0, 100.0 - wer)
            char_accuracy = max(0.0, 100.0 - cer)
        except Exception:
            wer, cer, word_accuracy, char_accuracy = 100.0, 100.0, 0.0, 0.0
        return wer, cer, word_accuracy, char_accuracy

    def calculate_wer(self, ground_truth: str, prediction: str) -> float:
        wer, _, _, _ = self.calculate_metrics(ground_truth, prediction)
        return wer

    def calculate_cer(self, ground_truth: str, prediction: str) -> float:
        _, cer, _, _ = self.calculate_metrics(ground_truth, prediction)
        return cer

    def calculate_accuracy(self, ground_truth: str, prediction: str) -> float:
        _, _, _, char_accuracy = self.calculate_metrics(ground_truth, prediction)
        return char_accuracy

    def calculate_word_accuracy(self, ground_truth: str, prediction: str) -> float:
        _, _, word_accuracy, _ = self.calculate_metrics(ground_truth, prediction)
        return word_accuracy

    # ----- Инференс -----
    def predict_text(self, image_path: str) -> str:
        try:
            # EasyOCR возвращает список детекций; берём полный конкатенированный текст
            results = self.reader.readtext(image_path, detail=0)
            if isinstance(results, list):
                text = " ".join([t for t in results if isinstance(t, str)]).strip()
            else:
                text = str(results).strip()
            return text
        except Exception as e:
            print(f"Ошибка EasyOCR при распознавании {image_path}: {e}")
            return ""

    # ----- Оценка датасета (интерфейс аналогичен TrOCREvaluator) -----
    def load_annotations(self, annotations_file: str, dataset_path: str) -> Dict[str, str]:
        annotations = {}
        try:
            with open(annotations_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        image_rel, text = parts
                        image_path = os.path.join(dataset_path, os.path.basename(image_rel))
                        annotations[image_path] = text
        except Exception as e:
            print(f"Ошибка загрузки аннотаций: {e}")
        return annotations

    def create_annotations_from_filenames(self, dataset_path: str) -> Dict[str, str]:
        annotations = {}
        for fname in os.listdir(dataset_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                annotations[os.path.join(dataset_path, fname)] = os.path.splitext(fname)[0]
        return annotations

    def evaluate_dataset(self, dataset_path: str, annotations_file: str = None, limit: int = 50) -> Dict:
        results = {
            'predictions': [],
            'ground_truths': [],
            'wer_scores': [],
            'cer_scores': [],
            'accuracy_scores': [],
            'word_accuracy_scores': []
        }

        if annotations_file and os.path.exists(annotations_file):
            annotations = self.load_annotations(annotations_file, dataset_path)
        else:
            annotations = self.create_annotations_from_filenames(dataset_path)

        items = list(annotations.items())[:limit] if limit else list(annotations.items())

        for image_path, gt_text in tqdm(items, desc="EasyOCR Evaluation"):
            pred = self.predict_text(image_path)
            wer, cer, wacc, cacc = self.calculate_metrics(gt_text, pred)
            results['predictions'].append(pred)
            results['ground_truths'].append(gt_text)
            results['wer_scores'].append(wer)
            results['cer_scores'].append(cer)
            results['accuracy_scores'].append(cacc)
            results['word_accuracy_scores'].append(wacc)

        return results

    # Совместимые методы для анализа/визуализации/сохранения
    def analyze_results(self, results: Dict) -> Dict:
        stats = {
            'total_samples': len(results['wer_scores']),
            'mean_wer': float(np.mean(results['wer_scores'])) if results['wer_scores'] else 0.0,
            'std_wer': float(np.std(results['wer_scores'])) if results['wer_scores'] else 0.0,
            'mean_cer': float(np.mean(results['cer_scores'])) if results['cer_scores'] else 0.0,
            'std_cer': float(np.std(results['cer_scores'])) if results['cer_scores'] else 0.0,
            'mean_accuracy': float(np.mean(results['accuracy_scores'])) if results['accuracy_scores'] else 0.0,
            'std_accuracy': float(np.std(results['accuracy_scores'])) if results['accuracy_scores'] else 0.0,
            'mean_word_accuracy': float(np.mean(results['word_accuracy_scores'])) if results['word_accuracy_scores'] else 0.0,
            'std_word_accuracy': float(np.std(results['word_accuracy_scores'])) if results['word_accuracy_scores'] else 0.0,
            'min_wer': float(np.min(results['wer_scores'])) if results['wer_scores'] else 0.0,
            'max_wer': float(np.max(results['wer_scores'])) if results['wer_scores'] else 0.0,
            'min_cer': float(np.min(results['cer_scores'])) if results['cer_scores'] else 0.0,
            'max_cer': float(np.max(results['cer_scores'])) if results['cer_scores'] else 0.0,
            'min_accuracy': float(np.min(results['accuracy_scores'])) if results['accuracy_scores'] else 0.0,
            'max_accuracy': float(np.max(results['accuracy_scores'])) if results['accuracy_scores'] else 0.0,
            'min_word_accuracy': float(np.min(results['word_accuracy_scores'])) if results['word_accuracy_scores'] else 0.0,
            'max_word_accuracy': float(np.max(results['word_accuracy_scores'])) if results['word_accuracy_scores'] else 0.0,
        }
        return stats

    def plot_results(self, results: Dict, save_path: str = None, show: bool = False):
        # идентично TrOCREvaluator: упрощённый вариант
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes[0, 0].hist(results['wer_scores'], bins=30, color='#6366f1', alpha=0.8)
        axes[0, 0].set_title('Распределение WER')
        axes[0, 1].hist(results['cer_scores'], bins=30, color='#ef4444', alpha=0.8)
        axes[0, 1].set_title('Распределение CER')
        axes[1, 0].scatter(results['wer_scores'], results['cer_scores'], color='#f59e0b', alpha=0.7)
        axes[1, 0].set_title('WER vs CER')
        axes[1, 1].boxplot([results['wer_scores'], results['cer_scores']], labels=['WER', 'CER'])
        axes[1, 1].set_title('Box Plot метрик')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)

    def save_results(self, results: Dict, stats: Dict, output_path: str = "evaluation_results.json"):
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
                'accuracy': results['accuracy_scores'][i],
                'word_accuracy': results['word_accuracy_scores'][i]
            })
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        df = pd.DataFrame({
            'ground_truth': results['ground_truths'],
            'prediction': results['predictions'],
            'wer': results['wer_scores'],
            'cer': results['cer_scores'],
            'accuracy': results['accuracy_scores'],
            'word_accuracy': results['word_accuracy_scores']
        })
        csv_path = output_path.replace('.json', '.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
