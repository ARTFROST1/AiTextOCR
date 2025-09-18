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
    QTableWidgetItem, QHeaderView, QCheckBox, QLineEdit, QTextBrowser,
    QSizePolicy, QScrollBar, QToolButton
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap, QIcon

import qdarkstyle
from PyQt5.QtCore import QPropertyAnimation, QEasingCurve, QRect

# Импорт нашей логики TrOCR
from trocr_evaluation import TrOCREvaluator


# Современная цветовая схема
class ModernColors:
    """Современная цветовая палитра для тёмной темы"""
    
    # Основные цвета
    PRIMARY = "#6366f1"      # Индиго
    PRIMARY_DARK = "#4f46e5" # Тёмный индиго
    PRIMARY_LIGHT = "#818cf8" # Светлый индиго
    
    # Акцентные цвета
    SUCCESS = "#10b981"      # Изумрудный
    SUCCESS_DARK = "#059669" # Тёмный изумрудный
    WARNING = "#f59e0b"      # Янтарный
    WARNING_DARK = "#d97706" # Тёмный янтарный
    ERROR = "#ef4444"        # Красный
    ERROR_DARK = "#dc2626"   # Тёмный красный
    INFO = "#3b82f6"         # Синий
    INFO_DARK = "#2563eb"    # Тёмный синий
    
    # Нейтральные цвета
    BACKGROUND = "#0f172a"   # Очень тёмный синий
    SURFACE = "#1e293b"      # Тёмный синий
    SURFACE_LIGHT = "#334155" # Светлый тёмный синий
    BORDER = "#475569"       # Серо-синий
    TEXT_PRIMARY = "#f8fafc" # Почти белый
    TEXT_SECONDARY = "#cbd5e1" # Светло-серый
    TEXT_MUTED = "#94a3b8"   # Серый
    
    # Градиенты
    GRADIENT_PRIMARY = "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6366f1, stop:1 #8b5cf6)"
    GRADIENT_SUCCESS = "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #10b981, stop:1 #059669)"
    GRADIENT_WARNING = "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f59e0b, stop:1 #d97706)"
    GRADIENT_ERROR = "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ef4444, stop:1 #dc2626)"


class ModernStyles:
    """Современные стили для компонентов"""
    
    @staticmethod
    def get_main_window_style():
        """Стиль главного окна"""
        return f"""
        QMainWindow {{
            background-color: {ModernColors.BACKGROUND};
            color: {ModernColors.TEXT_PRIMARY};
        }}
        
        QTabWidget::pane {{
            border: 1px solid {ModernColors.BORDER};
            border-radius: 12px;
            background-color: {ModernColors.SURFACE};
            margin: 8px;
        }}
        
        QTabBar::tab {{
            background-color: {ModernColors.SURFACE_LIGHT};
            color: {ModernColors.TEXT_SECONDARY};
            padding: 12px 24px;
            margin: 4px;
            border-radius: 8px;
            font-weight: 500;
            font-size: 14px;
        }}
        
        QTabBar::tab:selected {{
            background: {ModernColors.GRADIENT_PRIMARY};
            color: {ModernColors.TEXT_PRIMARY};
            font-weight: 600;
        }}
        
        QTabBar::tab:hover {{
            background-color: {ModernColors.SURFACE};
            color: {ModernColors.TEXT_PRIMARY};
        }}
        """
    
    @staticmethod
    def get_group_box_style():
        """Стиль групповых элементов"""
        return f"""
        QGroupBox {{
            font-weight: 600;
            font-size: 16px;
            color: {ModernColors.TEXT_PRIMARY};
            border: 2px solid {ModernColors.BORDER};
            border-radius: 12px;
            margin: 8px 0px;
            padding-top: 16px;
            background-color: {ModernColors.SURFACE};
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 16px;
            padding: 0 8px 0 8px;
            background-color: {ModernColors.SURFACE};
        }}
        """
    
    @staticmethod
    def get_button_style(button_type="primary"):
        """Стили кнопок"""
        if button_type == "primary":
            return f"""
            QPushButton {{
                background: {ModernColors.GRADIENT_PRIMARY};
                color: {ModernColors.TEXT_PRIMARY};
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                border-radius: 8px;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #7c3aed, stop:1 #6366f1);
            }}
            QPushButton:pressed {{
                background: {ModernColors.PRIMARY_DARK};
            }}
            QPushButton:disabled {{
                background-color: {ModernColors.SURFACE_LIGHT};
                color: {ModernColors.TEXT_MUTED};
            }}
            """
        elif button_type == "success":
            return f"""
            QPushButton {{
                background: {ModernColors.GRADIENT_SUCCESS};
                color: {ModernColors.TEXT_PRIMARY};
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                border-radius: 8px;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #059669, stop:1 #10b981);
            }}
            QPushButton:pressed {{
                background: {ModernColors.SUCCESS_DARK};
            }}
            """
        elif button_type == "warning":
            return f"""
            QPushButton {{
                background: {ModernColors.GRADIENT_WARNING};
                color: {ModernColors.TEXT_PRIMARY};
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                border-radius: 8px;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #d97706, stop:1 #f59e0b);
            }}
            QPushButton:pressed {{
                background: {ModernColors.WARNING_DARK};
            }}
            """
        elif button_type == "error":
            return f"""
            QPushButton {{
                background: {ModernColors.GRADIENT_ERROR};
                color: {ModernColors.TEXT_PRIMARY};
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                border-radius: 8px;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #dc2626, stop:1 #ef4444);
            }}
            QPushButton:pressed {{
                background: {ModernColors.ERROR_DARK};
            }}
            """
        elif button_type == "secondary":
            return f"""
            QPushButton {{
                background-color: {ModernColors.SURFACE_LIGHT};
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER};
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                border-radius: 8px;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {ModernColors.SURFACE};
                border-color: {ModernColors.PRIMARY};
            }}
            QPushButton:pressed {{
                background-color: {ModernColors.BORDER};
            }}
            """
    
    @staticmethod
    def get_input_style():
        """Стиль полей ввода"""
        return f"""
        QLineEdit, QSpinBox, QComboBox {{
            background-color: {ModernColors.SURFACE_LIGHT};
            color: {ModernColors.TEXT_PRIMARY};
            border: 2px solid {ModernColors.BORDER};
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 14px;
            min-height: 20px;
        }}
        
        QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
            border-color: {ModernColors.PRIMARY};
            background-color: {ModernColors.SURFACE};
        }}
        
        QComboBox::drop-down {{
            border: none;
            background-color: {ModernColors.PRIMARY};
            border-radius: 0 6px 6px 0;
            width: 20px;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid {ModernColors.TEXT_PRIMARY};
            margin-right: 5px;
        }}
        
        QComboBox QAbstractItemView {{
            background-color: {ModernColors.SURFACE};
            border: 1px solid {ModernColors.BORDER};
            border-radius: 8px;
            selection-background-color: {ModernColors.PRIMARY};
        }}
        """
    
    @staticmethod
    def get_progress_bar_style():
        """Стиль прогресс-бара"""
        return f"""
        QProgressBar {{
            border: 2px solid {ModernColors.BORDER};
            border-radius: 8px;
            text-align: center;
            background-color: {ModernColors.SURFACE_LIGHT};
            color: {ModernColors.TEXT_PRIMARY};
            font-weight: 600;
        }}
        
        QProgressBar::chunk {{
            background: {ModernColors.GRADIENT_PRIMARY};
            border-radius: 6px;
        }}
        """
    
    @staticmethod
    def get_table_style():
        """Стиль таблиц"""
        return f"""
        QTableWidget {{
            background-color: {ModernColors.SURFACE};
            color: {ModernColors.TEXT_PRIMARY};
            border: 1px solid {ModernColors.BORDER};
            border-radius: 8px;
            gridline-color: {ModernColors.BORDER};
            selection-background-color: {ModernColors.PRIMARY};
        }}
        
        QTableWidget::item {{
            padding: 8px;
            border-bottom: 1px solid {ModernColors.BORDER};
        }}
        
        QTableWidget::item:selected {{
            background-color: {ModernColors.PRIMARY};
            color: {ModernColors.TEXT_PRIMARY};
        }}
        
        QHeaderView::section {{
            background-color: {ModernColors.SURFACE_LIGHT};
            color: {ModernColors.TEXT_PRIMARY};
            padding: 8px;
            border: none;
            border-right: 1px solid {ModernColors.BORDER};
            font-weight: 600;
        }}
        
        QHeaderView::section:first {{
            border-top-left-radius: 8px;
        }}
        
        QHeaderView::section:last {{
            border-top-right-radius: 8px;
            border-right: none;
        }}
        """


class ConsoleOutput(QTextBrowser):
    """Виджет для отображения консольного вывода с современным дизайном"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_styling()
        
    def setup_ui(self):
        """Настройка интерфейса консоли"""
        # Настройка шрифта
        font = QFont("Consolas", 10)
        font.setStyleHint(QFont.Monospace)
        self.setFont(font)
        
        # Настройка поведения
        self.setReadOnly(True)
        self.setLineWrapMode(QTextBrowser.WidgetWidth)
        self.setOpenExternalLinks(False)
        
        # Настройка прокрутки
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
    def setup_styling(self):
        """Настройка стилей консоли"""
        self.setStyleSheet(f"""
            QTextBrowser {{
                background-color: {ModernColors.BACKGROUND};
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 8px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 11px;
                line-height: 1.4;
            }}
            
            QTextBrowser:focus {{
                border-color: {ModernColors.PRIMARY};
            }}
            
            QScrollBar:vertical {{
                background-color: {ModernColors.SURFACE_LIGHT};
                width: 12px;
                border-radius: 6px;
            }}
            
            QScrollBar::handle:vertical {{
                background-color: {ModernColors.BORDER};
                border-radius: 6px;
                min-height: 20px;
            }}
            
            QScrollBar::handle:vertical:hover {{
                background-color: {ModernColors.PRIMARY};
            }}
        """)
    
    def append_text(self, text, color=None):
        """Добавление текста с возможностью цветового выделения"""
        if color is None:
            color = ModernColors.TEXT_PRIMARY
            
        # Улучшенное форматирование с подсветкой синтаксиса
        formatted_text = self.format_console_text(text, color)
        self.append(formatted_text)
        
        # Автоскролл к концу
        self.moveCursor(self.textCursor().End)
        
    def format_console_text(self, text, color):
        """Форматирование текста консоли с подсветкой"""
        # Определяем тип сообщения и применяем соответствующее форматирование
        if "🚀" in text and "ЗАПУСК" in text:
            # Заголовок запуска
            return f'<div style="color: {ModernColors.PRIMARY}; font-weight: bold; margin: 4px 0px;">{text}</div>'
        elif "📊" in text and "РЕЗУЛЬТАТЫ" in text:
            # Заголовок результатов
            return f'<div style="color: {ModernColors.SUCCESS}; font-weight: bold; margin: 4px 0px;">{text}</div>'
        elif "🔄" in text and "Прогресс" in text:
            # Прогресс
            return f'<div style="color: {ModernColors.INFO}; margin: 2px 0px;">{text}</div>'
        elif "✅" in text or "успешно" in text:
            # Успех
            return f'<div style="color: {ModernColors.SUCCESS}; margin: 2px 0px;">{text}</div>'
        elif "❌" in text or "Ошибка" in text:
            # Ошибка
            return f'<div style="color: {ModernColors.ERROR}; margin: 2px 0px;">{text}</div>'
        elif "⚠️" in text or "Предупреждение" in text:
            # Предупреждение
            return f'<div style="color: {ModernColors.WARNING}; margin: 2px 0px;">{text}</div>'
        elif "=" in text and len(text) > 20:
            # Разделители
            return f'<div style="color: {ModernColors.BORDER}; margin: 2px 0px;">{text}</div>'
        else:
            # Обычный текст
            return f'<div style="color: {color}; margin: 1px 0px;">{text}</div>'
        
    def clear_console(self):
        """Очистка консоли"""
        self.clear()
        
    def get_console_text(self):
        """Получение текста консоли для копирования"""
        return self.toPlainText()


class ConsoleWidget(QWidget):
    """Контейнер для консольного окна с управлением"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Настройка интерфейса консольного виджета"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Заголовок консоли
        header_layout = QHBoxLayout()
        
        # Иконка и заголовок
        title_label = QLabel("🖥️ Консоль")
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_PRIMARY};
                font-size: 14px;
                font-weight: 600;
                padding: 4px 0px;
            }}
        """)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Кнопки управления
        self.clear_btn = QToolButton()
        self.clear_btn.setText("🗑️")
        self.clear_btn.setToolTip("Очистить консоль")
        self.clear_btn.setStyleSheet(f"""
            QToolButton {{
                background-color: {ModernColors.SURFACE_LIGHT};
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
            }}
            QToolButton:hover {{
                background-color: {ModernColors.SURFACE};
                border-color: {ModernColors.ERROR};
            }}
        """)
        
        self.copy_btn = QToolButton()
        self.copy_btn.setText("📋")
        self.copy_btn.setToolTip("Копировать в буфер обмена")
        self.copy_btn.setStyleSheet(f"""
            QToolButton {{
                background-color: {ModernColors.SURFACE_LIGHT};
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
            }}
            QToolButton:hover {{
                background-color: {ModernColors.SURFACE};
                border-color: {ModernColors.PRIMARY};
            }}
        """)
        
        self.toggle_btn = QToolButton()
        self.toggle_btn.setText("📉")
        self.toggle_btn.setToolTip("Свернуть/развернуть консоль")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setStyleSheet(f"""
            QToolButton {{
                background-color: {ModernColors.SURFACE_LIGHT};
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
            }}
            QToolButton:hover {{
                background-color: {ModernColors.SURFACE};
                border-color: {ModernColors.PRIMARY};
            }}
            QToolButton:checked {{
                background-color: {ModernColors.PRIMARY};
                color: {ModernColors.TEXT_PRIMARY};
            }}
        """)
        
        # Кнопка настроек консоли
        self.settings_btn = QToolButton()
        self.settings_btn.setText("⚙️")
        self.settings_btn.setToolTip("Настройки консоли")
        self.settings_btn.setStyleSheet(f"""
            QToolButton {{
                background-color: {ModernColors.SURFACE_LIGHT};
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
            }}
            QToolButton:hover {{
                background-color: {ModernColors.SURFACE};
                border-color: {ModernColors.PRIMARY};
            }}
        """)
        
        header_layout.addWidget(self.clear_btn)
        header_layout.addWidget(self.copy_btn)
        header_layout.addWidget(self.settings_btn)
        header_layout.addWidget(self.toggle_btn)
        
        layout.addLayout(header_layout)
        
        # Консольное окно
        self.console = ConsoleOutput()
        self.console.setMaximumHeight(200)
        self.console.setMinimumHeight(100)
        layout.addWidget(self.console)
        
        # Изначально консоль свернута
        self.is_collapsed = True
        self.console.setVisible(False)
        self.toggle_btn.setChecked(False)
        
    def setup_connections(self):
        """Настройка соединений"""
        self.clear_btn.clicked.connect(self.clear_console)
        self.copy_btn.clicked.connect(self.copy_to_clipboard)
        self.toggle_btn.clicked.connect(self.toggle_console)
        self.settings_btn.clicked.connect(self.show_console_settings)
        
    def clear_console(self):
        """Очистка консоли"""
        self.console.clear_console()
        
    def copy_to_clipboard(self):
        """Копирование в буфер обмена"""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.console.get_console_text())
        
    def toggle_console(self):
        """Переключение видимости консоли с анимацией"""
        self.is_collapsed = not self.is_collapsed
        
        # Создаем анимацию для плавного изменения размера
        if not hasattr(self, 'console_animation'):
            self.console_animation = QPropertyAnimation(self.console, b"maximumHeight")
            self.console_animation.setDuration(300)
            self.console_animation.setEasingCurve(QEasingCurve.OutCubic)
        
        if self.is_collapsed:
            # Сворачиваем консоль
            self.console_animation.setStartValue(200)
            self.console_animation.setEndValue(0)
            self.console.setVisible(False)
            self.toggle_btn.setText("📈")
            self.toggle_btn.setToolTip("Развернуть консоль")
        else:
            # Разворачиваем консоль
            self.console.setVisible(True)
            self.console_animation.setStartValue(0)
            self.console_animation.setEndValue(200)
            self.toggle_btn.setText("📉")
            self.toggle_btn.setToolTip("Свернуть консоль")
            
        self.console_animation.start()
            
    def append_text(self, text, color=None):
        """Добавление текста в консоль"""
        self.console.append_text(text, color)
        
    def append_success(self, text):
        """Добавление успешного сообщения"""
        self.append_text(text, ModernColors.SUCCESS)
        
    def append_error(self, text):
        """Добавление сообщения об ошибке"""
        self.append_text(text, ModernColors.ERROR)
        
    def append_warning(self, text):
        """Добавление предупреждения"""
        self.append_text(text, ModernColors.WARNING)
        
    def append_info(self, text):
        """Добавление информационного сообщения"""
        self.append_text(text, ModernColors.INFO)
        
    def show_console_settings(self):
        """Показ настроек консоли"""
        dialog = QMessageBox()
        dialog.setWindowTitle("Настройки консоли")
        dialog.setText("Настройки консоли:")
        dialog.setInformativeText(
            "• Размер шрифта: 11px\n"
            "• Шрифт: Consolas (моноширинный)\n"
            "• Тема: Тёмная\n"
            "• Автоскролл: Включён\n"
            "• Подсветка синтаксиса: Включена"
        )
        dialog.setIcon(QMessageBox.Information)
        dialog.exec_()


class PrintCapture:
    """Класс для перехвата print() вызовов и дублирования в GUI"""
    
    def __init__(self, console_widget):
        self.console_widget = console_widget
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def write(self, text):
        """Перехват вывода"""
        # Выводим в оригинальный stdout
        self.original_stdout.write(text)
        self.original_stdout.flush()
        
        # Дублируем в GUI консоль
        if text.strip():  # Игнорируем пустые строки
            # Определяем тип сообщения по содержимому
            if "❌" in text or "Ошибка" in text:
                self.console_widget.append_error(text.strip())
            elif "✅" in text or "успешно" in text:
                self.console_widget.append_success(text.strip())
            elif "⚠️" in text or "Предупреждение" in text:
                self.console_widget.append_warning(text.strip())
            elif "🔄" in text or "Прогресс" in text:
                self.console_widget.append_info(text.strip())
            else:
                self.console_widget.append_text(text.strip())
    
    def flush(self):
        """Сброс буфера"""
        self.original_stdout.flush()


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
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Заголовок графика
        header_label = QLabel("📈 Визуализация результатов")
        header_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_PRIMARY};
                font-size: 16px;
                font-weight: 600;
                padding: 8px 0px;
            }}
        """)
        layout.addWidget(header_label)
        
        # Создаем matplotlib canvas с современными настройками
        self.figure = Figure(figsize=(12, 8), facecolor=ModernColors.SURFACE)
        self.canvas = FigureCanvas(self.figure)
        
        # Настройка стиля matplotlib для тёмной темы
        plt.style.use('dark_background')
        
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def plot_results(self, results: Dict):
        """Построение графиков результатов"""
        self.figure.clear()
        
        # Настройка цветовой схемы
        colors = {
            'primary': ModernColors.PRIMARY,
            'success': ModernColors.SUCCESS,
            'warning': ModernColors.WARNING,
            'error': ModernColors.ERROR,
            'text': ModernColors.TEXT_PRIMARY,
            'text_secondary': ModernColors.TEXT_SECONDARY,
            'surface': ModernColors.SURFACE,
            'border': ModernColors.BORDER
        }
        
        # Создаем 2x2 сетку графиков
        axes = self.figure.subplots(2, 2)
        
        # Настройка общего стиля
        for ax in axes.flat:
            ax.set_facecolor(ModernColors.SURFACE_LIGHT)
            ax.tick_params(colors=colors['text_secondary'])
            ax.spines['bottom'].set_color(colors['border'])
            ax.spines['top'].set_color(colors['border'])
            ax.spines['right'].set_color(colors['border'])
            ax.spines['left'].set_color(colors['border'])
        
        # Распределение WER
        axes[0, 0].hist(results['wer_scores'], bins=30, alpha=0.8, 
                       color=colors['primary'], edgecolor=colors['border'], linewidth=1)
        axes[0, 0].set_title('Распределение WER', fontsize=14, fontweight='bold', color=colors['text'])
        axes[0, 0].set_xlabel('WER (%)', color=colors['text_secondary'])
        axes[0, 0].set_ylabel('Частота', color=colors['text_secondary'])
        axes[0, 0].grid(True, alpha=0.3, color=colors['border'])
        
        # Распределение CER
        axes[0, 1].hist(results['cer_scores'], bins=30, alpha=0.8, 
                       color=colors['error'], edgecolor=colors['border'], linewidth=1)
        axes[0, 1].set_title('Распределение CER', fontsize=14, fontweight='bold', color=colors['text'])
        axes[0, 1].set_xlabel('CER (%)', color=colors['text_secondary'])
        axes[0, 1].set_ylabel('Частота', color=colors['text_secondary'])
        axes[0, 1].grid(True, alpha=0.3, color=colors['border'])
        
        # Сравнение WER и CER
        scatter = axes[1, 0].scatter(results['wer_scores'], results['cer_scores'], 
                                   alpha=0.7, color=colors['warning'], s=60, 
                                   edgecolors=colors['border'], linewidth=0.5)
        axes[1, 0].set_title('WER vs CER', fontsize=14, fontweight='bold', color=colors['text'])
        axes[1, 0].set_xlabel('WER (%)', color=colors['text_secondary'])
        axes[1, 0].set_ylabel('CER (%)', color=colors['text_secondary'])
        axes[1, 0].grid(True, alpha=0.3, color=colors['border'])
        
        # Box plot метрик
        data_for_box = [results['wer_scores'], results['cer_scores']]
        bp = axes[1, 1].boxplot(data_for_box, labels=['WER', 'CER'], patch_artist=True)
        bp['boxes'][0].set_facecolor(colors['primary'])
        bp['boxes'][0].set_alpha(0.8)
        bp['boxes'][1].set_facecolor(colors['error'])
        bp['boxes'][1].set_alpha(0.8)
        
        # Стилизация элементов box plot
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[element], color=colors['border'])
        
        axes[1, 1].set_title('Распределение метрик', fontsize=14, fontweight='bold', color=colors['text'])
        axes[1, 1].set_ylabel('Процент ошибок', color=colors['text_secondary'])
        axes[1, 1].grid(True, alpha=0.3, color=colors['border'])
        
        # Настройка общего стиля фигуры
        self.figure.patch.set_facecolor(ModernColors.SURFACE)
        
        self.figure.tight_layout(pad=2.0)
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
        self.console_widget = None  # Консольное окно
        self.print_capture = None  # Перехватчик print()
        self.profiles_dir = "profiles"  # Папка для профилей
        self.settings_file = os.path.join(self.profiles_dir, "trocr_settings.json")  # Файл настроек по умолчанию
        
        self.setup_ui()
        self.setup_connections()
        self.setup_console_capture()
        self.setup_profiles_directory()  # Создаем папку для профилей
        # Загружаем настройки после создания всех элементов
        QTimer.singleShot(100, self.load_default_settings)
        
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        self.setWindowTitle("TrOCR Evaluation Tool - Modern UI")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Применяем современные стили
        self.setStyleSheet(ModernStyles.get_main_window_style())
        
        # Центральный виджет с вкладками
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Главный layout с отступами
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)
        
        # Создаем вкладки
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(ModernStyles.get_main_window_style())
        main_layout.addWidget(self.tab_widget)
        
        # Вкладка настроек
        self.setup_settings_tab()
        
        # Вкладка результатов
        self.setup_results_tab()
        
        # Вкладка детальных результатов
        self.setup_details_tab()
        
        # Статус бар с современным стилем
        self.statusBar().setStyleSheet(f"""
            QStatusBar {{
                background-color: {ModernColors.SURFACE};
                color: {ModernColors.TEXT_SECONDARY};
                border-top: 1px solid {ModernColors.BORDER};
                padding: 8px;
            }}
        """)
        self.statusBar().showMessage("Готов к работе")
        
        # Добавляем анимацию появления окна
        self.setup_animations()
    
    def setup_animations(self):
        """Настройка анимаций для плавных переходов"""
        # Анимация появления окна
        self.fade_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_animation.setDuration(300)
        self.fade_animation.setStartValue(0.0)
        self.fade_animation.setEndValue(1.0)
        self.fade_animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # Анимация для прогресс-бара
        self.progress_animation = QPropertyAnimation(self.progress_bar, b"value")
        self.progress_animation.setDuration(200)
        self.progress_animation.setEasingCurve(QEasingCurve.OutQuad)
        
    def showEvent(self, event):
        """Обработчик события показа окна"""
        super().showEvent(event)
        # Запускаем анимацию появления
        self.fade_animation.start()
        
    def closeEvent(self, event):
        """Обработчик события закрытия окна"""
        # Автоматически сохраняем настройки при закрытии
        self.auto_save_settings()
        # Восстанавливаем оригинальный stdout
        self.restore_console()
        super().closeEvent(event)
    
    def setup_profiles_directory(self):
        """Создание папки для профилей"""
        try:
            os.makedirs(self.profiles_dir, exist_ok=True)
            print(f"📁 Папка профилей создана: {self.profiles_dir}")
        except Exception as e:
            print(f"⚠️ Не удалось создать папку профилей: {e}")
    
    def setup_console_capture(self):
        """Настройка перехвата print() вызовов"""
        if self.console_widget:
            self.print_capture = PrintCapture(self.console_widget)
            # Перехватываем stdout для дублирования в GUI
            sys.stdout = self.print_capture
            
    def restore_console(self):
        """Восстановление оригинального stdout"""
        if self.print_capture:
            sys.stdout = self.print_capture.original_stdout
    
    def load_default_settings(self):
        """Автоматическая загрузка настроек при запуске"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                self.apply_settings(settings)
                print(f"✅ Настройки загружены из {self.settings_file}")
            except Exception as e:
                print(f"⚠️ Не удалось загрузить настройки: {e}")
    
    def apply_settings(self, settings):
        """Применение настроек к интерфейсу"""
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
    
    def auto_save_settings(self):
        """Автоматическое сохранение настроек"""
        settings = {
            'model_name': self.model_combo.currentText(),
            'dataset_path': self.dataset_path_edit.text(),
            'annotations_path': self.annotations_path_edit.text(),
            'limit': self.limit_spinbox.value()
        }
        
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Не удалось сохранить настройки: {e}")
        
    def setup_settings_tab(self):
        """Настройка вкладки с параметрами"""
        settings_widget = QWidget()
        layout = QVBoxLayout(settings_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(20)
        
        # Группа выбора модели
        model_group = QGroupBox("🤖 Выбор модели")
        model_group.setStyleSheet(ModernStyles.get_group_box_style())
        model_layout = QGridLayout(model_group)
        model_layout.setSpacing(12)
        model_layout.setContentsMargins(20, 20, 20, 20)
        
        # Заголовок модели
        model_label = QLabel("🤖 Модель TrOCR:")
        model_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_PRIMARY};
                font-size: 14px;
                font-weight: 600;
            }}
        """)
        model_layout.addWidget(model_label, 0, 0)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "microsoft/trocr-base-handwritten",
            "microsoft/trocr-large-handwritten"
        ])
        self.model_combo.setStyleSheet(ModernStyles.get_input_style())
        model_layout.addWidget(self.model_combo, 0, 1)
        
        # Информация о модели
        self.model_info_label = QLabel("Base модель: быстрее, меньше размер")
        self.model_info_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_MUTED};
                font-style: italic;
                font-size: 12px;
                padding: 4px 0px;
            }}
        """)
        model_layout.addWidget(self.model_info_label, 1, 1)
        
        # Обновляем информацию при изменении модели
        self.model_combo.currentTextChanged.connect(self.update_model_info)
        
        layout.addWidget(model_group)
        
        # Группа выбора датасета
        dataset_group = QGroupBox("📁 Настройки датасета")
        dataset_group.setStyleSheet(ModernStyles.get_group_box_style())
        dataset_layout = QGridLayout(dataset_group)
        dataset_layout.setSpacing(12)
        dataset_layout.setContentsMargins(20, 20, 20, 20)
        
        # Путь к изображениям
        dataset_label = QLabel("Путь к изображениям:")
        dataset_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_PRIMARY};
                font-size: 14px;
                font-weight: 600;
            }}
        """)
        dataset_layout.addWidget(dataset_label, 0, 0)
        
        self.dataset_path_edit = QLineEdit("IAM/image")
        self.dataset_path_edit.setStyleSheet(ModernStyles.get_input_style())
        dataset_layout.addWidget(self.dataset_path_edit, 0, 1)
        
        self.dataset_browse_btn = QPushButton("📂 Обзор...")
        self.dataset_browse_btn.setStyleSheet(ModernStyles.get_button_style("secondary"))
        dataset_layout.addWidget(self.dataset_browse_btn, 0, 2)
        
        # Файл аннотаций
        annotations_label = QLabel("Файл аннотаций:")
        annotations_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_PRIMARY};
                font-size: 14px;
                font-weight: 600;
            }}
        """)
        dataset_layout.addWidget(annotations_label, 1, 0)
        
        self.annotations_path_edit = QLineEdit("IAM/gt_test.txt")
        self.annotations_path_edit.setStyleSheet(ModernStyles.get_input_style())
        dataset_layout.addWidget(self.annotations_path_edit, 1, 1)
        
        self.annotations_browse_btn = QPushButton("📄 Обзор...")
        self.annotations_browse_btn.setStyleSheet(ModernStyles.get_button_style("secondary"))
        dataset_layout.addWidget(self.annotations_browse_btn, 1, 2)
        
        # Количество изображений
        limit_label = QLabel("Количество изображений:")
        limit_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_PRIMARY};
                font-size: 14px;
                font-weight: 600;
            }}
        """)
        dataset_layout.addWidget(limit_label, 2, 0)
        
        self.limit_spinbox = QSpinBox()
        self.limit_spinbox.setRange(1, 1000)
        self.limit_spinbox.setValue(50)
        self.limit_spinbox.setStyleSheet(ModernStyles.get_input_style())
        dataset_layout.addWidget(self.limit_spinbox, 2, 1)
        
        layout.addWidget(dataset_group)
        
        # Группа управления
        control_group = QGroupBox("🎮 Управление")
        control_group.setStyleSheet(ModernStyles.get_group_box_style())
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(16)
        control_layout.setContentsMargins(20, 20, 20, 20)
        
        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(ModernStyles.get_progress_bar_style())
        control_layout.addWidget(self.progress_bar)
        
        # Статус
        self.status_label = QLabel("Готов к запуску")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_SECONDARY};
                font-size: 14px;
                font-weight: 500;
                padding: 8px 0px;
            }}
        """)
        control_layout.addWidget(self.status_label)
        
        # Кнопки в сетке
        button_layout = QGridLayout()
        button_layout.setSpacing(12)
        
        # Большая главная кнопка запуска/остановки
        self.main_action_btn = QPushButton("🚀 Запустить оценку")
        self.main_action_btn.setStyleSheet(ModernStyles.get_button_style("success"))
        self.main_action_btn.setMinimumHeight(50)  # Делаем кнопку выше
        self.main_action_btn.setFont(QFont("Arial", 14, QFont.Bold))  # Больший шрифт
        button_layout.addWidget(self.main_action_btn, 0, 0, 1, 2)  # Занимает 2 колонки
        
        self.save_settings_btn = QPushButton("💾 Сохранить профиль")
        self.save_settings_btn.setStyleSheet(ModernStyles.get_button_style("primary"))
        self.save_settings_btn.setToolTip("Сохранить текущие настройки как профиль")
        button_layout.addWidget(self.save_settings_btn, 1, 0)
        
        self.load_settings_btn = QPushButton("📁 Загрузить профиль")
        self.load_settings_btn.setStyleSheet(ModernStyles.get_button_style("primary"))
        self.load_settings_btn.setToolTip("Загрузить сохраненный профиль настроек")
        button_layout.addWidget(self.load_settings_btn, 1, 1)
        
        self.open_general_results_btn = QPushButton("📂 Открыть папку results")
        self.open_general_results_btn.setStyleSheet(ModernStyles.get_button_style("warning"))
        button_layout.addWidget(self.open_general_results_btn, 2, 0, 1, 2)
        
        control_layout.addLayout(button_layout)
        layout.addWidget(control_group)
        
        # Консольное окно
        self.console_widget = ConsoleWidget()
        layout.addWidget(self.console_widget)
        
        layout.addStretch()
        self.tab_widget.addTab(settings_widget, "🏠 Главная")
    
    def setup_results_tab(self):
        """Настройка вкладки с результатами"""
        results_widget = QWidget()
        layout = QVBoxLayout(results_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Создаем splitter для разделения на графики и метрики
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {ModernColors.BORDER};
                width: 2px;
            }}
            QSplitter::handle:hover {{
                background-color: {ModernColors.PRIMARY};
            }}
        """)
        
        # Левая панель - графики
        self.plot_widget = ResultsPlotWidget()
        splitter.addWidget(self.plot_widget)
        
        # Правая панель - метрики
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        metrics_layout.setContentsMargins(16, 16, 16, 16)
        metrics_layout.setSpacing(16)
        
        # Группа основных метрик
        metrics_group = QGroupBox("📊 Основные метрики")
        metrics_group.setStyleSheet(ModernStyles.get_group_box_style())
        metrics_group_layout = QGridLayout(metrics_group)
        metrics_group_layout.setSpacing(8)
        metrics_group_layout.setContentsMargins(20, 20, 20, 20)
        
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
            # Лейбл названия метрики
            label = QLabel(label_text)
            label.setStyleSheet(f"""
                QLabel {{
                    color: {ModernColors.TEXT_SECONDARY};
                    font-size: 13px;
                    font-weight: 500;
                }}
            """)
            metrics_group_layout.addWidget(label, i, 0)
            
            # Лейбл значения метрики
            value_label = QLabel("—")
            value_label.setStyleSheet(f"""
                QLabel {{
                    font-weight: 700;
                    color: {ModernColors.PRIMARY};
                    font-size: 14px;
                    background-color: {ModernColors.SURFACE_LIGHT};
                    padding: 4px 8px;
                    border-radius: 4px;
                    border: 1px solid {ModernColors.BORDER};
                }}
            """)
            self.metrics_labels[key] = value_label
            metrics_group_layout.addWidget(value_label, i, 1)
        
        metrics_layout.addWidget(metrics_group)
        
        # Группа оценки качества
        quality_group = QGroupBox("🎯 Оценка качества")
        quality_group.setStyleSheet(ModernStyles.get_group_box_style())
        quality_layout = QVBoxLayout(quality_group)
        quality_layout.setContentsMargins(20, 20, 20, 20)
        quality_layout.setSpacing(12)
        
        self.quality_label = QLabel("Запустите оценку для получения результатов")
        self.quality_label.setStyleSheet(f"""
            QLabel {{
                font-size: 14px;
                padding: 12px;
                color: {ModernColors.TEXT_SECONDARY};
                background-color: {ModernColors.SURFACE_LIGHT};
                border-radius: 8px;
                border: 1px solid {ModernColors.BORDER};
            }}
        """)
        self.quality_label.setWordWrap(True)
        quality_layout.addWidget(self.quality_label)
        
        metrics_layout.addWidget(quality_group)
        
        # Кнопка открытия папки с результатами
        self.open_results_btn = QPushButton("📂 Открыть папку с результатами")
        self.open_results_btn.setEnabled(False)
        self.open_results_btn.setStyleSheet(ModernStyles.get_button_style("error"))
        
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
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Заголовок вкладки
        header_label = QLabel("📋 Детальные результаты оценки")
        header_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_PRIMARY};
                font-size: 18px;
                font-weight: 700;
                padding: 8px 0px;
            }}
        """)
        layout.addWidget(header_label)
        
        # Таблица с детальными результатами
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Эталонный текст", "Распознанный текст", "WER (%)", "CER (%)", "Точность (символы) (%)", "Точность (слова) (%)"
        ])
        
        # Применяем современные стили к таблице
        self.results_table.setStyleSheet(ModernStyles.get_table_style())
        
        # Настройка таблицы
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        
        # Дополнительные настройки таблицы
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSortingEnabled(True)
        
        layout.addWidget(self.results_table)
        self.tab_widget.addTab(details_widget, "📋 Детали")
    
    def setup_connections(self):
        """Настройка соединений сигналов и слотов"""
        self.main_action_btn.clicked.connect(self.toggle_evaluation)
        self.dataset_browse_btn.clicked.connect(self.browse_dataset)
        self.annotations_browse_btn.clicked.connect(self.browse_annotations)
        self.save_settings_btn.clicked.connect(self.save_settings)
        self.load_settings_btn.clicked.connect(self.load_settings)
        self.open_general_results_btn.clicked.connect(self.open_general_results_folder)
        self.open_results_btn.clicked.connect(self.open_results_folder)
        
        # Автоматическое сохранение при изменении настроек
        if hasattr(self, 'model_combo'):
            self.model_combo.currentTextChanged.connect(self.auto_save_settings)
        if hasattr(self, 'dataset_path_edit'):
            self.dataset_path_edit.textChanged.connect(self.auto_save_settings)
        if hasattr(self, 'annotations_path_edit'):
            self.annotations_path_edit.textChanged.connect(self.auto_save_settings)
        if hasattr(self, 'limit_spinbox'):
            self.limit_spinbox.valueChanged.connect(self.auto_save_settings)
    
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
    
    def toggle_evaluation(self):
        """Переключение между запуском и остановкой оценки"""
        if self.worker_thread and self.worker_thread.isRunning():
            # Если процесс запущен, останавливаем его
            self.stop_evaluation()
        else:
            # Если процесс не запущен, запускаем его
            self.start_evaluation()
    
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
            self.main_action_btn.setText("⏹️ Остановить")
            self.main_action_btn.setStyleSheet(ModernStyles.get_button_style("error"))
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Создаем папку для результатов
            self.current_results_folder = self.create_results_folder(model_name, dataset_path)
            
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
        """Обновление прогресса с анимацией"""
        # Анимированное обновление прогресс-бара
        self.progress_animation.setStartValue(self.progress_bar.value())
        self.progress_animation.setEndValue(value)
        self.progress_animation.start()
        
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
        self.main_action_btn.setText("🚀 Запустить оценку")
        self.main_action_btn.setStyleSheet(ModernStyles.get_button_style("success"))
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
            'limit': self.limit_spinbox.value(),
            'saved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'profile_name': f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        # Предлагаем сохранить в папку профилей
        default_filename = f"trocr_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        default_path = os.path.join(self.profiles_dir, default_filename)
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Сохранить профиль настроек", 
            default_path, 
            "JSON files (*.json)"
        )
        
        if file_path:
            try:
                # Сохраняем в папку профилей
                if not file_path.startswith(self.profiles_dir):
                    filename = os.path.basename(file_path)
                    file_path = os.path.join(self.profiles_dir, filename)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(settings, f, ensure_ascii=False, indent=2)
                
                QMessageBox.information(
                    self, 
                    "✅ Профиль сохранен", 
                    f"Профиль настроек успешно сохранен в:\n{file_path}\n\n"
                    f"Модель: {settings['model_name']}\n"
                    f"Датасет: {settings['dataset_path']}\n"
                    f"Изображений: {settings['limit']}\n"
                    f"Сохранен: {settings['saved_at']}"
                )
                print(f"💾 Профиль настроек сохранен: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "❌ Ошибка", f"Ошибка при сохранении профиля:\n{str(e)}")
                print(f"❌ Ошибка сохранения профиля: {e}")
    
    def load_settings(self):
        """Загрузка настроек из файла"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Загрузить профиль настроек", 
            self.profiles_dir,  # Начинаем с папки профилей
            "JSON files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                # Применяем настройки
                self.apply_settings(settings)
                
                # Показываем информацию о загруженном профиле
                saved_at = settings.get('saved_at', 'Неизвестно')
                QMessageBox.information(
                    self, 
                    "✅ Профиль загружен", 
                    f"Профиль настроек успешно загружен из:\n{file_path}\n\n"
                    f"Модель: {settings.get('model_name', 'Не указана')}\n"
                    f"Датасет: {settings.get('dataset_path', 'Не указан')}\n"
                    f"Изображений: {settings.get('limit', 'Не указано')}\n"
                    f"Сохранен: {saved_at}"
                )
                print(f"📁 Профиль настроек загружен: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "❌ Ошибка", f"Ошибка при загрузке профиля:\n{str(e)}")
                print(f"❌ Ошибка загрузки профиля: {e}")
    
    def create_results_folder(self, model_name: str, dataset_path: str = None) -> str:
        """
        Создание папки для результатов с датой/временем, названием модели и датасета
        
        Args:
            model_name: Название модели
            dataset_path: Путь к датасету (опционально)
            
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
        
        # Извлекаем название датасета из пути
        dataset_short = "unknown"
        if dataset_path:
            # Нормализуем путь
            normalized_path = dataset_path.rstrip('/\\')
            
            # Всегда берем родительскую папку как название датасета
            # Это работает независимо от названия папки с изображениями
            dataset_short = os.path.basename(os.path.dirname(normalized_path))
            
            # Специальные случаи для известных датасетов
            if dataset_short.lower() in ['iam']:
                dataset_short = "IAM"
        
        # Создаем название папки
        folder_name = f"{timestamp}_{dataset_short}_{model_short}"
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
    
    # Применяем современную темную тему
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    # Дополнительные стили для приложения
    app.setStyleSheet(app.styleSheet() + f"""
        QApplication {{
            background-color: {ModernColors.BACKGROUND};
            color: {ModernColors.TEXT_PRIMARY};
        }}
        
        QScrollBar:vertical {{
            background-color: {ModernColors.SURFACE_LIGHT};
            width: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {ModernColors.BORDER};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {ModernColors.PRIMARY};
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        
        QScrollBar:horizontal {{
            background-color: {ModernColors.SURFACE_LIGHT};
            height: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:horizontal {{
            background-color: {ModernColors.BORDER};
            border-radius: 6px;
            min-width: 20px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background-color: {ModernColors.PRIMARY};
        }}
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
    """)
    
    # Создаем и показываем главное окно
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
