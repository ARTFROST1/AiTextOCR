"""
TrOCR GUI Application
Красивое графическое приложение для оценки моделей TrOCR
"""

import sys
import os
import json
import time
import threading
import subprocess
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QTabWidget, QLabel, QPushButton, QComboBox, 
    QSpinBox, QProgressBar, QTextEdit, QFileDialog, QMessageBox,
    QGroupBox, QSplitter, QFrame, QScrollArea, QTableWidget, 
    QTableWidgetItem, QHeaderView, QCheckBox, QLineEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap, QIcon

import qdarkstyle

# Импорт нашей логики TrOCR
from trocr_evaluation import TrOCREvaluator


class EvaluationWorker(QThread):
    """Worker thread для выполнения оценки в фоне"""
    progress_updated = pyqtSignal(int, str)  # progress, status_text
    evaluation_completed = pyqtSignal(dict, dict)  # results, stats
    error_occurred = pyqtSignal(str)
    
    def __init__(self, evaluator, dataset_path, annotations_file, limit, results_folder):
        super().__init__()
        self.evaluator = evaluator
        self.dataset_path = dataset_path
        self.annotations_file = annotations_file
        self.limit = limit
        self.results_folder = results_folder
        self.is_running = True
    
    def run(self):
        try:
            self.progress_updated.emit(10, "Инициализация модели...")
            
            if not self.is_running:
                return
                
            # Оценка модели
            self.progress_updated.emit(30, "Начинается обработка изображений...")
            results = self.evaluator.evaluate_dataset(
                self.dataset_path, 
                self.annotations_file, 
                limit=self.limit
            )
            
            if not self.is_running:
                return
                
            self.progress_updated.emit(80, "Анализ результатов...")
            
            # Анализ результатов
            stats = self.evaluator.analyze_results(results)
            
            if not self.is_running:
                return
                
            self.progress_updated.emit(90, "Сохранение результатов...")
            
            # Сохранение результатов в новую папку
            self.save_results_to_folder(results, stats)
            
            if not self.is_running:
                return
                
            self.progress_updated.emit(100, "Оценка завершена!")
            
            # Эмитируем результаты
            self.evaluation_completed.emit(results, stats)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def save_results_to_folder(self, results, stats):
        """Сохранение результатов в папку"""
        try:
            # Пути к файлам в новой папке
            json_path = os.path.join(self.results_folder, "evaluation_results.json")
            csv_path = os.path.join(self.results_folder, "evaluation_results.csv")
            png_path = os.path.join(self.results_folder, "evaluation_plots.png")
            
            # Сохраняем результаты
            self.evaluator.save_results(results, stats, json_path)
            
            # Создаем графики
            self.evaluator.plot_results(results, png_path, show=False)
            
            print(f"💾 Результаты сохранены в папку: {self.results_folder}")
            
        except Exception as e:
            print(f"❌ Ошибка при сохранении результатов: {e}")
            raise e
    
    def stop(self):
        self.is_running = False


class ResultsPlotWidget(QWidget):
    """Виджет для отображения графиков результатов"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Создаем matplotlib canvas
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def plot_results(self, results: Dict):
        """Построение графиков результатов"""
        self.figure.clear()
        
        # Создаем 2x2 сетку графиков
        axes = self.figure.subplots(2, 2)
        
        # Распределение WER
        axes[0, 0].hist(results['wer_scores'], bins=30, alpha=0.7, color='#3498db', edgecolor='black')
        axes[0, 0].set_title('Распределение WER', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('WER (%)')
        axes[0, 0].set_ylabel('Частота')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Распределение CER
        axes[0, 1].hist(results['cer_scores'], bins=30, alpha=0.7, color='#e74c3c', edgecolor='black')
        axes[0, 1].set_title('Распределение CER', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('CER (%)')
        axes[0, 1].set_ylabel('Частота')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Сравнение WER и CER
        axes[1, 0].scatter(results['wer_scores'], results['cer_scores'], 
                          alpha=0.6, color='#9b59b6', s=50)
        axes[1, 0].set_title('WER vs CER', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('WER (%)')
        axes[1, 0].set_ylabel('CER (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot метрик
        data_for_box = [results['wer_scores'], results['cer_scores']]
        bp = axes[1, 1].boxplot(data_for_box, labels=['WER', 'CER'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][1].set_facecolor('#e74c3c')
        axes[1, 1].set_title('Распределение метрик', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Процент ошибок')
        axes[1, 1].grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()


class MainWindow(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        self.evaluator = None
        self.current_results = None
        self.current_stats = None
        self.worker_thread = None
        self.start_time = None  # Время начала оценки
        self.current_results_folder = None  # Текущая папка с результатами
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        self.setWindowTitle("TrOCR Evaluation Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Центральный виджет с вкладками
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Главный layout
        main_layout = QVBoxLayout(central_widget)
        
        # Создаем вкладки
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Вкладка настроек
        self.setup_settings_tab()
        
        # Вкладка результатов
        self.setup_results_tab()
        
        # Вкладка детальных результатов
        self.setup_details_tab()
        
        # Статус бар
        self.statusBar().showMessage("Готов к работе")
        
    def setup_settings_tab(self):
        """Настройка вкладки с параметрами"""
        settings_widget = QWidget()
        layout = QVBoxLayout(settings_widget)
        
        # Группа выбора модели
        model_group = QGroupBox("Выбор модели")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("Модель TrOCR:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "microsoft/trocr-base-handwritten",
            "microsoft/trocr-large-handwritten"
        ])
        model_layout.addWidget(self.model_combo, 0, 1)
        
        # Информация о модели
        self.model_info_label = QLabel("Base модель: быстрее, меньше размер")
        self.model_info_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        model_layout.addWidget(self.model_info_label, 1, 1)
        
        # Обновляем информацию при изменении модели
        self.model_combo.currentTextChanged.connect(self.update_model_info)
        
        layout.addWidget(model_group)
        
        # Группа выбора датасета
        dataset_group = QGroupBox("Настройки датасета")
        dataset_layout = QGridLayout(dataset_group)
        
        dataset_layout.addWidget(QLabel("Путь к изображениям:"), 0, 0)
        self.dataset_path_edit = QLineEdit("IAM/image")
        self.dataset_browse_btn = QPushButton("Обзор...")
        dataset_layout.addWidget(self.dataset_path_edit, 0, 1)
        dataset_layout.addWidget(self.dataset_browse_btn, 0, 2)
        
        dataset_layout.addWidget(QLabel("Файл аннотаций:"), 1, 0)
        self.annotations_path_edit = QLineEdit("IAM/gt_test.txt")
        self.annotations_browse_btn = QPushButton("Обзор...")
        dataset_layout.addWidget(self.annotations_path_edit, 1, 1)
        dataset_layout.addWidget(self.annotations_browse_btn, 1, 2)
        
        dataset_layout.addWidget(QLabel("Количество изображений:"), 2, 0)
        self.limit_spinbox = QSpinBox()
        self.limit_spinbox.setRange(1, 1000)
        self.limit_spinbox.setValue(50)
        dataset_layout.addWidget(self.limit_spinbox, 2, 1)
        
        layout.addWidget(dataset_group)
        
        # Группа управления
        control_group = QGroupBox("Управление")
        control_layout = QVBoxLayout(control_group)
        
        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)
        
        # Статус
        self.status_label = QLabel("Готов к запуску")
        control_layout.addWidget(self.status_label)
        
        # Кнопки
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("🚀 Запустить оценку")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:pressed {
                background-color: #229954;
            }
        """)
        
        self.stop_btn = QPushButton("⏹️ Остановить")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #ec7063;
            }
        """)
        
        self.save_settings_btn = QPushButton("💾 Сохранить настройки")
        self.save_settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #5dade2;
            }
        """)
        
        self.load_settings_btn = QPushButton("📁 Загрузить настройки")
        self.load_settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #bb8fce;
            }
        """)
        
        self.open_general_results_btn = QPushButton("📂 Открыть общую папку results")
        self.open_general_results_btn.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #f7dc6f;
            }
        """)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.save_settings_btn)
        button_layout.addWidget(self.load_settings_btn)
        button_layout.addWidget(self.open_general_results_btn)
        button_layout.addStretch()
        
        control_layout.addLayout(button_layout)
        layout.addWidget(control_group)
        
        layout.addStretch()
        self.tab_widget.addTab(settings_widget, "⚙️ Настройки")
    
    def setup_results_tab(self):
        """Настройка вкладки с результатами"""
        results_widget = QWidget()
        layout = QVBoxLayout(results_widget)
        
        # Создаем splitter для разделения на графики и метрики
        splitter = QSplitter(Qt.Horizontal)
        
        # Левая панель - графики
        self.plot_widget = ResultsPlotWidget()
        splitter.addWidget(self.plot_widget)
        
        # Правая панель - метрики
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        
        # Группа основных метрик
        metrics_group = QGroupBox("Основные метрики")
        metrics_group_layout = QGridLayout(metrics_group)
        
        # Создаем лейблы для метрик
        self.metrics_labels = {}
        metrics = [
            ("total_samples", "Образцов обработано:"),
            ("mean_wer", "Средний WER:"),
            ("mean_cer", "Средний CER:"),
            ("mean_accuracy", "Средняя точность (символы):"),
            ("mean_word_accuracy", "Средняя точность (слова):"),
            ("std_wer", "Стд. отклонение WER:"),
            ("std_cer", "Стд. отклонение CER:"),
            ("std_accuracy", "Стд. отклонение точности:"),
            ("min_wer", "Минимальный WER:"),
            ("max_wer", "Максимальный WER:"),
            ("min_accuracy", "Минимальная точность:"),
            ("max_accuracy", "Максимальная точность:")
        ]
        
        for i, (key, label_text) in enumerate(metrics):
            metrics_group_layout.addWidget(QLabel(label_text), i, 0)
            value_label = QLabel("—")
            value_label.setStyleSheet("font-weight: bold; color: #3498db;")
            self.metrics_labels[key] = value_label
            metrics_group_layout.addWidget(value_label, i, 1)
        
        metrics_layout.addWidget(metrics_group)
        
        # Группа оценки качества
        quality_group = QGroupBox("Оценка качества")
        quality_layout = QVBoxLayout(quality_group)
        
        self.quality_label = QLabel("Запустите оценку для получения результатов")
        self.quality_label.setStyleSheet("font-size: 14px; padding: 10px;")
        self.quality_label.setWordWrap(True)
        quality_layout.addWidget(self.quality_label)
        
        metrics_layout.addWidget(quality_group)
        
        # Кнопка открытия папки с результатами
        self.open_results_btn = QPushButton("📂 Открыть папку с результатами")
        self.open_results_btn.setEnabled(False)
        self.open_results_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 15px 25px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #ec7063;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        
        metrics_layout.addWidget(self.open_results_btn)
        metrics_layout.addStretch()
        
        splitter.addWidget(metrics_widget)
        splitter.setSizes([800, 400])
        
        layout.addWidget(splitter)
        self.tab_widget.addTab(results_widget, "📊 Результаты")
    
    def setup_details_tab(self):
        """Настройка вкладки с детальными результатами"""
        details_widget = QWidget()
        layout = QVBoxLayout(details_widget)
        
        # Таблица с детальными результатами
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Эталонный текст", "Распознанный текст", "WER (%)", "CER (%)", "Точность (символы) (%)", "Точность (слова) (%)"
        ])
        
        # Настройка таблицы
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        
        layout.addWidget(self.results_table)
        self.tab_widget.addTab(details_widget, "📋 Детали")
    
    def setup_connections(self):
        """Настройка соединений сигналов и слотов"""
        self.start_btn.clicked.connect(self.start_evaluation)
        self.stop_btn.clicked.connect(self.stop_evaluation)
        self.dataset_browse_btn.clicked.connect(self.browse_dataset)
        self.annotations_browse_btn.clicked.connect(self.browse_annotations)
        self.save_settings_btn.clicked.connect(self.save_settings)
        self.load_settings_btn.clicked.connect(self.load_settings)
        self.open_general_results_btn.clicked.connect(self.open_general_results_folder)
        self.open_results_btn.clicked.connect(self.open_results_folder)
    
    def update_model_info(self, model_name):
        """Обновление информации о выбранной модели"""
        if "base" in model_name:
            self.model_info_label.setText("Base модель: быстрее, меньше размер")
        elif "large" in model_name:
            self.model_info_label.setText("Large модель: точнее, больше размер")
        else:
            self.model_info_label.setText("")
    
    def browse_dataset(self):
        """Выбор папки с изображениями"""
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if folder:
            self.dataset_path_edit.setText(folder)
    
    def browse_annotations(self):
        """Выбор файла аннотаций"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Выберите файл аннотаций", 
            "", 
            "Text files (*.txt);;JSON files (*.json)"
        )
        if file_path:
            self.annotations_path_edit.setText(file_path)
    
    def start_evaluation(self):
        """Запуск оценки"""
        try:
            # Получаем параметры
            model_name = self.model_combo.currentText()
            dataset_path = self.dataset_path_edit.text()
            annotations_file = self.annotations_path_edit.text()
            limit = self.limit_spinbox.value()
            
            # Проверяем пути
            if not os.path.exists(dataset_path):
                QMessageBox.warning(self, "Ошибка", f"Папка с изображениями не найдена: {dataset_path}")
                return
            
            if not os.path.exists(annotations_file):
                QMessageBox.warning(self, "Ошибка", f"Файл аннотаций не найден: {annotations_file}")
                return
            
            # Инициализируем оценщик
            self.status_label.setText("Инициализация модели...")
            self.evaluator = TrOCREvaluator(model_name)
            
            # Настраиваем UI
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Создаем папку для результатов
            self.current_results_folder = self.create_results_folder(model_name)
            
            # Запоминаем время начала
            import time
            self.start_time = time.time()
            
            # Выводим информацию о начале оценки в консоль
            print("\n" + "="*60)
            print("🚀 ЗАПУСК ОЦЕНКИ МОДЕЛИ TrOCR")
            print("="*60)
            print(f"📊 Модель: {model_name}")
            print(f"📁 Датасет: {dataset_path}")
            print(f"📄 Аннотации: {annotations_file}")
            print(f"🔢 Количество изображений: {limit}")
            print(f"📂 Папка результатов: {self.current_results_folder}")
            print("="*60)
            
            # Запускаем worker thread
            self.worker_thread = EvaluationWorker(
                self.evaluator, dataset_path, annotations_file, limit, self.current_results_folder
            )
            self.worker_thread.progress_updated.connect(self.update_progress)
            self.worker_thread.evaluation_completed.connect(self.evaluation_finished)
            self.worker_thread.error_occurred.connect(self.evaluation_error)
            self.worker_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при запуске: {str(e)}")
            self.reset_ui()
    
    def stop_evaluation(self):
        """Остановка оценки"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait()
        
        self.reset_ui()
        self.status_label.setText("Оценка остановлена")
    
    def update_progress(self, value, status_text):
        """Обновление прогресса"""
        self.progress_bar.setValue(value)
        self.status_label.setText(status_text)
        self.statusBar().showMessage(status_text)
        
        # Выводим прогресс в консоль
        print(f"🔄 Прогресс: {value}% - {status_text}")
    
    def evaluation_finished(self, results, stats):
        """Обработка завершения оценки"""
        self.current_results = results
        self.current_stats = stats
        
        # Обновляем метрики
        self.update_metrics(stats)
        
        # Обновляем графики
        self.plot_widget.plot_results(results)
        
        # Обновляем таблицу
        self.update_results_table(results)
        
        # Переключаемся на вкладку результатов
        self.tab_widget.setCurrentIndex(1)
        
        self.reset_ui()
        self.status_label.setText("Оценка завершена успешно!")
        
        # Активируем кнопку открытия папки с результатами
        self.open_results_btn.setEnabled(True)
        
        # Выводим результаты в консоль (как в оригинальном скрипте)
        self.print_results_to_console(results, stats)
        
        # Показываем оценку качества
        self.show_quality_assessment(stats)
    
    def evaluation_error(self, error_message):
        """Обработка ошибки оценки"""
        QMessageBox.critical(self, "Ошибка", f"Ошибка при оценке: {error_message}")
        self.reset_ui()
        self.status_label.setText("Ошибка при оценке")
    
    def reset_ui(self):
        """Сброс UI в исходное состояние"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        # Не деактивируем кнопку открытия папки, чтобы можно было открыть последние результаты
    
    def update_metrics(self, stats):
        """Обновление отображения метрик"""
        for key, label in self.metrics_labels.items():
            if key in stats:
                if key == 'total_samples':
                    label.setText(str(stats[key]))
                else:
                    label.setText(f"{stats[key]:.2f}%")
    
    def update_results_table(self, results):
        """Обновление таблицы с результатами"""
        num_rows = len(results['predictions'])
        self.results_table.setRowCount(num_rows)
        
        for i in range(num_rows):
            self.results_table.setItem(i, 0, QTableWidgetItem(results['ground_truths'][i]))
            self.results_table.setItem(i, 1, QTableWidgetItem(results['predictions'][i]))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{results['wer_scores'][i]:.2f}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{results['cer_scores'][i]:.2f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{results['accuracy_scores'][i]:.2f}"))
            self.results_table.setItem(i, 5, QTableWidgetItem(f"{results['word_accuracy_scores'][i]:.2f}"))
    
    def show_quality_assessment(self, stats):
        """Показ оценки качества"""
        mean_wer = stats['mean_wer']
        
        if mean_wer < 20:
            quality_text = "🟢 <b>Отличное качество распознавания!</b><br>Модель показывает превосходные результаты."
            color = "#27ae60"
        elif mean_wer < 40:
            quality_text = "🟡 <b>Хорошее качество распознавания</b><br>Модель работает хорошо, но есть возможности для улучшения."
            color = "#f39c12"
        elif mean_wer < 60:
            quality_text = "🟠 <b>Удовлетворительное качество</b><br>Модель работает приемлемо, но требует доработки."
            color = "#e67e22"
        else:
            quality_text = "🔴 <b>Низкое качество распознавания</b><br>Модель требует значительного улучшения или другой подход."
            color = "#e74c3c"
        
        self.quality_label.setText(quality_text)
        self.quality_label.setStyleSheet(f"font-size: 14px; padding: 10px; color: {color};")
    
    def print_results_to_console(self, results, stats):
        """Вывод результатов в консоль (как в оригинальном скрипте)"""
        print("\n" + "="*60)
        print("📊 РЕЗУЛЬТАТЫ ОЦЕНКИ")
        print("="*60)
        
        # Красивое отображение статистики
        print(f"📈 Обработано образцов: {stats['total_samples']}")
        print("\n")
        print(f"📉 Средний WER:        {stats['mean_wer']:.2f}%")
        print(f"📈 Средний CER:        {stats['mean_cer']:.2f}%")
        print(f"📊 Средняя точность (символы):   {stats['mean_accuracy']:.2f}%")
        print(f"📊 Средняя точность (слова):     {stats['mean_word_accuracy']:.2f}%")
        print("\n")
        print(f"📉 Стандартное отклонение WER: {stats['std_wer']:.2f}%")
        print(f"📈 Стандартное отклонение CER: {stats['std_cer']:.2f}%")
        print(f"📊 Стандартное отклонение точности: {stats['std_accuracy']:.2f}%")
        
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
        
        # Сравнение WER и CER
        if stats['mean_wer'] > 0:
            print(f"📈 CER в {stats['mean_cer'] / stats['mean_wer']:.1f} раз лучше WER")
        
        # Дополнительная статистика
        print(f"\n📊 ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА:")
        print(f"📉 Минимальный WER: {stats['min_wer']:.2f}%")
        print(f"📉 Максимальный WER: {stats['max_wer']:.2f}%")
        print(f"📊 Минимальная точность: {stats['min_accuracy']:.2f}%")
        print(f"📊 Максимальная точность: {stats['max_accuracy']:.2f}%")
        
        # Время выполнения
        if self.start_time:
            import time
            end_time = time.time()
            duration = end_time - self.start_time
            print(f"\n⏱️  Время выполнения: {duration:.1f} секунд")
        
        # Информация о файлах
        print(f"\n💾 СОЗДАННЫЕ ФАЙЛЫ:")
        if self.current_results_folder:
            print(f"📂 Папка результатов: {self.current_results_folder}")
            print(f"📄 JSON: {os.path.join(self.current_results_folder, 'evaluation_results.json')}")
            print(f"📊 CSV:  {os.path.join(self.current_results_folder, 'evaluation_results.csv')}")
            print(f"📈 PNG:  {os.path.join(self.current_results_folder, 'evaluation_plots.png')}")
        else:
            print(f"📄 JSON: results/trocr_real_iam_evaluation.json")
            print(f"📊 CSV:  results/trocr_real_iam_evaluation.csv")
            print(f"📈 PNG:  results/evaluation_real_plots.png")
        
        print("\n" + "="*60)
        print("✅ ОЦЕНКА ЗАВЕРШЕНА УСПЕШНО!")
        print("="*60)
    
    def save_settings(self):
        """Сохранение настроек в файл"""
        settings = {
            'model_name': self.model_combo.currentText(),
            'dataset_path': self.dataset_path_edit.text(),
            'annotations_path': self.annotations_path_edit.text(),
            'limit': self.limit_spinbox.value()
        }
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Сохранить настройки", 
            "trocr_settings.json", 
            "JSON files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(settings, f, ensure_ascii=False, indent=2)
                QMessageBox.information(self, "Успех", f"Настройки сохранены в {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении: {str(e)}")
    
    def load_settings(self):
        """Загрузка настроек из файла"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Загрузить настройки", 
            "", 
            "JSON files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                # Применяем настройки
                if 'model_name' in settings:
                    index = self.model_combo.findText(settings['model_name'])
                    if index >= 0:
                        self.model_combo.setCurrentIndex(index)
                
                if 'dataset_path' in settings:
                    self.dataset_path_edit.setText(settings['dataset_path'])
                
                if 'annotations_path' in settings:
                    self.annotations_path_edit.setText(settings['annotations_path'])
                
                if 'limit' in settings:
                    self.limit_spinbox.setValue(settings['limit'])
                
                QMessageBox.information(self, "Успех", f"Настройки загружены из {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке: {str(e)}")
    
    def create_results_folder(self, model_name: str) -> str:
        """
        Создание папки для результатов с датой/временем и названием модели
        
        Args:
            model_name: Название модели
            
        Returns:
            Путь к созданной папке
        """
        # Создаем базовую папку results если её нет
        results_base = Path("results")
        results_base.mkdir(exist_ok=True)
        
        # Получаем текущую дату и время
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Извлекаем короткое название модели
        model_short = model_name.split("/")[-1] if "/" in model_name else model_name
        model_short = model_short.replace("microsoft-", "").replace("trocr-", "")
        
        # Создаем название папки
        folder_name = f"{timestamp}_{model_short}"
        folder_path = results_base / folder_name
        
        # Создаем папку
        folder_path.mkdir(exist_ok=True)
        
        print(f"📁 Создана папка результатов: {folder_path}")
        return str(folder_path)
    
    def open_results_folder(self):
        """Открытие папки с результатами в проводнике"""
        if not self.current_results_folder:
            QMessageBox.warning(self, "Предупреждение", "Нет результатов для открытия")
            return
        
        try:
            folder_path = Path(self.current_results_folder)
            if not folder_path.exists():
                QMessageBox.warning(self, "Ошибка", f"Папка не найдена: {folder_path}")
                return
            
            # Открываем папку в проводнике в зависимости от ОС
            if platform.system() == "Windows":
                os.startfile(str(folder_path))
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(folder_path)])
            else:  # Linux
                subprocess.run(["xdg-open", str(folder_path)])
            
            print(f"📂 Открыта папка с результатами: {folder_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось открыть папку: {str(e)}")
            print(f"❌ Ошибка при открытии папки: {e}")
    
    def open_general_results_folder(self):
        """Открытие общей папки results"""
        try:
            # Создаем папку results если её нет
            results_path = Path("results")
            results_path.mkdir(exist_ok=True)
            
            # Открываем папку в проводнике в зависимости от ОС
            if platform.system() == "Windows":
                os.startfile(str(results_path))
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(results_path)])
            else:  # Linux
                subprocess.run(["xdg-open", str(results_path)])
            
            print(f"📂 Открыта общая папка results: {results_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось открыть папку results: {str(e)}")
            print(f"❌ Ошибка при открытии папки results: {e}")


def main():
    """Главная функция"""
    app = QApplication(sys.argv)
    
    # Применяем темную тему
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    # Создаем и показываем главное окно
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
