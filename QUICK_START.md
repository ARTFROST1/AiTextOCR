# Быстрый старт (короткая инструкция)

Ниже — минимальные шаги, чтобы после клонирования репозитория запустить GUI и тесты.

## 1) Клонировать репозиторий
```powershell
git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ>.git
cd AiTextOCR
```

## 2) Установить окружение

```bash
python scripts/setup_environment.py
```

## 3) Активировать виртуальное окружение
```powershell
.\venv_cuda\Scripts\Activate.ps1
```

## 4) Запустить GUI
```powershell
python run_gui.py
```

Консольная оценка (чисто IAM на trOCR base 50 шт):
```powershell
python scripts\run_full_evaluation.py
```

## Полезно
- Результаты сохраняются в папке `results/`
