[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Предсказание статуса возврата Samsung

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
```
.
├── data
│   └── raw                     # исходный samsung_global_sales_dataset.csv
├── notebooks
│   ├── cp1_notebook.ipynb      # Ноутбук CP1
│   └── cp2_improvements.ipynb  # Ноутбук CP2
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
| Logistic Regression (baseline) | 0.8589 | 0.308 | только числовые признаки |
| RandomForest | 0.8589 | 0.308 | class_weight='balanced' |
| XGBoost | 0.5696 | 0.318 | sample_weight balanced |
| LightGBM | 0.6201 | 0.313 | class_weight='balanced' |
| CatBoost | 0.6804 | 0.329 | auto_class_weights='Balanced' |
| Tuned XGBoost (GridSearch) | 0.8589 | 0.308 | cv=3, f1_macro |
| XGBoost + PCA | 0.8589 | 0.308 | 644 компоненты (95% дисперсии) |
| Voting (RF+XGB+LGBM) | 0.8589 | 0.308 | soft voting |
| Stacking (RF+XGB+LGBM→LR) | 0.8589 | 0.308 | |
| **CatBoost** | **0.686** | **0.333** | лучший F1-macro |

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
