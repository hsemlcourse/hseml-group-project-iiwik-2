# Отчёт по проекту

**Студент:** Павлюк Алёна Романовна
**Группа:** БИВ231

---

## 1. Введение и постановка задачи

**Задача:** предсказать, что покупатель сделает с товаром после покупки - оставит, вернёт или обменяет.
Целевая переменная return_status принимает три значения:
- Kept - товар оставлен
- Returned - возвращён
- Exchanged - обменян

Классы сильно несбалансированы: Kept составляет 86% всех записей, Returned — 9%, Exchanged — 5%. В такой ситуации accuracy была бы обманчивой: модель, которая всегда предсказывает Kept, получает accuracy = 0.86, но совершенно бесполезна. F1-macro считает F1-меру по каждому классу отдельно и усредняет, поэтому F1-macro выбрана как основная метрика
---

## 2. Поиск и описание данных

**Источник:** Kaggle — [Samsung Global Product Sales Dataset](https://www.kaggle.com/datasets/ashyou09/samsung-global-product-sales-dataset/data)

**Описание датасета:**
- 15 500 строк, 28 столбцов
- Данные охватывают продажи продуктов Samsung по 20+ странам
- Каждая строка — одна транзакция с информацией о товаре, покупателе, канале продаж и статусе возврата

**Основные признаки:**

| Тип | Примеры признаков |
|-----|------------------|
| Числовые | `unit_price_usd`, `discount_pct`, `units_sold`, `revenue_usd`, `customer_rating` |
| Категориальные | `country`, `category`, `sales_channel`, `payment_method`, `previous_device_os` |
| Временны́е | `sale_date`, `year`, `quarter`, `month` |
| Целевая | `return_status` |

**Распределение целевой переменной:**

| Класс | Доля |
|-------|------|
| Kept | 86% |
| Returned | 9% |
| Exchanged | 5% |


---

## 3. Обработка и подготовка данных

**Качество данных:**
- Пропущенных значений — нет
- Дублей — нет
- Выбросов, требующих удаления, — не обнаружено

**Feature engineering** — создано 4 новых признака:

| Признак | Формула / Логика |
|---------|-----------------|
| `discount_amount` | `unit_price_usd × discount_pct / 100` — абсолютная сумма скидки |
| `revenue_per_unit` | `revenue_usd / units_sold` — выручка на единицу товара |
| `is_high_discount` | `1` если `discount_pct > 10`, иначе `0` |
| `month_num` | числовой номер месяца из `sale_date` |

Итого признаков для обучения: 29 (25 исходных − `sale_id`, `sale_date`, `return_status` + 4 новых).

**Сплит данных:**

| Выборка | Размер |
|---------|--------|
| Train | 80% (12 192 строки) |
| Validation | 10% (1 524 строки) |
| Test | 10% (1 524 строки) |

Использован `stratify=y` — пропорции классов сохранены во всех выборках. Data leak исключён: `fit` делается только на train, `transform` применяется к val/test.

**Предобработка:**
- Числовые признаки: `passthrough` (для CatBoost) или `StandardScaler` (для линейных моделей)
- Категориальные признаки: `OneHotEncoder(handle_unknown='ignore')` или нативная обработка CatBoost
- Для балансировки классов использовались `class_weight='balanced'` / `auto_class_weights='Balanced'` / `sample_weight`

---

## 4. Baseline-модель

В качестве baseline взята **логистическая регрессия** — простая линейная модель без feature engineering, только на числовых признаках со стандартизацией.

```
Logistic Regression: Accuracy = 0.686, F1-macro = 0.333
```

**Classification report:**

| Класс | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Exchanged | 0.05 | 0.07 | 0.06 |
| Kept | 0.86 | 0.78 | 0.82 |
| Returned | 0.1 | 0.17 | 0.13 |

**Вывод:** baseline полностью игнорирует редкие классы — предсказывает только `Kept`. Accuracy высокая, но F1-macro = 0.333 отражает реальную бесполезность модели для задачи.

---

## 5. Эксперименты

Все эксперименты проводились на validation set. Финальные метрики финальной модели проверены на test set.

| # | Гипотеза | Как проверялось | Модель | Accuracy | F1-macro |
|---|----------|----------------|--------|----------|----------|
| 1 | Baseline: простая линейная модель | LogReg, только числовые признаки, StandardScaler | Logistic Regression | 0.859 | 0.308 |
| 2 | Деревья лучше справятся с дисбалансом | RandomForest с `class_weight='balanced'`, все признаки через OHE | RandomForest | 0.859 | 0.308 |
| 3 | Бустинг даст прирост по F1 | XGBoost с `sample_weight` для балансировки | XGBoost | 0.570 | 0.318 |
| 4 | LightGBM быстрее и точнее XGBoost | LightGBM с `class_weight='balanced'` | LightGBM | 0.620 | 0.313 |
| 5 | CatBoost нативно работает с категориями | CatBoost с `auto_class_weights='Balanced'`, категориальные признаки без OHE | CatBoost | 0.680 | 0.329 |
| 6 | Подбор гиперпараметров улучшит XGBoost | GridSearchCV, cv=3, метрика f1_macro | Tuned XGBoost | 0.859 | 0.308 |
| 7 | PCA снизит шум и поможет моделям | PCA до 644 компонент (95% дисперсии) + XGBoost | XGBoost + PCA | 0.859 | 0.308 |
| 8 | Ансамбль даст синергию трёх моделей | Soft voting: RF + XGBoost + LightGBM | Voting Ensemble | 0.859 | 0.308 |
| 9 | Стекинг лучше простого голосования | RF + XGBoost + LightGBM → LogReg мета-модель | Stacking | 0.859 | 0.308 |
| 10 | CatBoost + полный препроцессинг — лучший результат | CatBoost с feature engineering, все признаки | **CatBoost (финал)** | **0.686** | **0.333** |

**Ключевые наблюдения:**
- Большинство моделей застревают в ловушке: предсказывают только `Kept` и получают accuracy ~0.859
- XGBoost и LightGBM немного лучше справляются с редкими классами (F1-macro 0.313–0.318), но ценой падения accuracy
- CatBoost показал наилучший баланс: F1-macro = 0.333 при разумной accuracy = 0.686
- PCA не помог — редукция размерности не улучшила качество
- Ансамблевые методы не дали прироста по сравнению с лучшей одиночной моделью

---

## 6. Финальная модель и интерпретируемость

**Выбор:** CatBoost с `auto_class_weights='Balanced'`

**Почему CatBoost:**
- Нативная обработка категориальных признаков — не нужен OHE, меньше утечек информации
- Встроенная балансировка классов
- Лучший F1-macro среди всех экспериментов (0.333 vs 0.308 у большинства конкурентов)

**Метрики на test set:**

| Метрика | Значение |
|---------|---------|
| Accuracy | 0.686 |
| F1-macro | 0.333 |

**Интерпретация результатов:**

Датасет синтетический — `return_status` слабо зависит от имеющихся признаков (цена, скидка, страна, канал продаж). Корреляция числовых признаков с целевой переменной близка к нулю. Это объясняет потолок качества ~0.333 по F1-macro: даже лучшая модель не может извлечь закономерности там, где их нет в данных.

**Feature importance (по RandomForest на числовых признаках):**  
Наиболее значимые признаки: `revenue_usd`, `unit_price_usd`, `revenue_per_unit`, `discount_amount`. Признаки `customer_rating` и `month_num` имеют минимальный вклад.

---

## 7. Деплой

**Стек деплоя:**
- **FastAPI** — REST API с эндпоинтами `/predict`, `/predict_batch`, `/health`
- **Streamlit** — веб-интерфейс для предсказаний без кода
- **Docker + docker-compose** — контейнеризация обоих сервисов

**Структура репозитория:**

```
.
├── app/
│   ├── main.py              # FastAPI сервер
│   └── streamlit_app.py     # Streamlit интерфейс
├── src/
│   └── train.py             # Обучение модели
├── models/
│   └── catboost_pipeline.pkl  # Сохранённая модель
├── notebooks/
│   ├── cp1-notebook.ipynb
│   └── cp2_improvements.ipynb
├── data/raw/
│   └── samsung_global_sales_dataset.csv
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
└── README.md
```

**Запуск:**

```bash
docker-compose up --build
```

- API: `http://localhost:8000/docs`
- Streamlit: `http://localhost:8501`

**Скриншот интерфейса Streamlit:**  
*лежит в папке screenshots*

**Скриншот FastAPI Swagger UI:**  
*лежит в папке screenshots*

**Видео работы приложения:**  
[https://drive.google.com/file/d/1wfRctKui9hdWi6oxZoqZX-VKFN0rWhk9/view?usp=sharing](https://drive.google.com/file/d/1wfRctKui9hdWi6oxZoqZX-VKFN0rWhk9/view?usp=sharing)


---

## 8. Заключение и выводы

**Что сделано:**
- Проведён EDA датасета (15 500 строк, 28 признаков)
- Создано 4 новых признака через feature engineering
- Обучено и сравнено 9 моделей + ансамблей
- Финальная модель — CatBoost с F1-macro = 0.333 на тест-выборке
- Развёрнуто REST API (FastAPI) + веб-интерфейс (Streamlit) в Docker

**Главный вывод:**  
Датасет синтетический, и `return_status` практически не зависит от доступных признаков — все модели упираются в потолок ~0.333 по F1-macro. В реальной задаче потребовались бы дополнительные данные: история возвратов конкретного покупателя, описание товара, отзывы, поведенческие признаки.

**Что можно улучшить:**
- Добавить реальные данные с осмысленной связью признаков и целевой переменной
- Попробовать SMOTE или другие методы oversamplig для редких классов
- Применить threshold tuning для управления балансом precision/recall по классам

