[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — [Название проекта]

**Студент:** Павлюк Алёна Романовна

**Группа:** БИВ231


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

<!-- Кратко опишите задачу: что предсказываем, какой датасет, метрика качества -->

**Задача:** Многоклассовая классификация - предсказание статуса возврата товара `return_status`. 
Классы:
- `Kept` — товар оставлен
- `Returned` — возвращён
- `Exchanged` — обменян

**Датасет:** Samsung Global Product Sales Dataset📱
https://www.kaggle.com/datasets/ashyou09/samsung-global-product-sales-dataset/data

15 500 строк, 28 колонок

**Целевая метрика:** `F1-macro` - среднее арифметическое F1-меры по каждому классу


## Структура репозитория
Опишите структуру проекта, сохранив при этом верхнеуровневые папки. Можно добавить новые при необходимости.
```
.
├── data
│   └── raw                     # исходный samsung_global_sales.csv
├── models                      # Сохранённые модели 
├── notebooks
│   ├── cp1_notebook.ipynb      # Основной ноутбук
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── preprocessing.py        # Предобработка данных
│   └── modeling.py             # Обучение и оценка моделей
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

Этот блок замените способом запуска вашего сервиса.
```bash
# 1. Клонировать репозиторий
git clone <url>
cd <repo-name>

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt
```

## Данные
- `data/raw/` — исходные файлы


## Результаты

| Модель | Accuracy | F1-macro | Примечание |
|--------|----------|----------|-------------|
| Logistic Regression (baseline) | 0.858710 | 0.307995 | только числовые признаки |
| RandomForest | 0.858710 | 0.307995 | class_weight='balanced' не дал улучшения |
| XGBoost | 0.576774 | 0.309178 | со взвешиванием выборки |


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
