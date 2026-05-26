import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent.parent / "models" / "catboost_pipeline.pkl"

app = FastAPI(
    title="Samsung Return Status Predictor",
    description="Предсказывает статус возврата товара: Kept / Returned / Exchanged",
    version="1.0.0",
)

with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
preprocessor = artifact["preprocessor"]
label_encoder = artifact["label_encoder"]
numeric_features = artifact["numeric_features"]
categorical_features = artifact["categorical_features"]


class SaleInput(BaseModel):

    year: int = Field(2023, description="Год продажи")
    unit_price_usd: float = Field(999.0, description="Цена за единицу в USD")
    discount_pct: int = Field(5, description="Процент скидки (0-100)")
    units_sold: int = Field(1, description="Количество единиц")
    discounted_price_usd: float = Field(949.05, description="Цена со скидкой в USD")
    revenue_usd: float = Field(949.05, description="Выручка в USD")
    fx_rate_to_usd: float = Field(1.0, description="Курс валюты к USD")
    revenue_local_currency: float = Field(949.05, description="Выручка в местной валюте")
    customer_rating: float = Field(4.0, description="Рейтинг покупателя (1-5)")
    discount_amount: float = Field(49.95, description="Сумма скидки в USD")
    revenue_per_unit: float = Field(949.05, description="Выручка на единицу товара")
    is_high_discount: int = Field(0, description="Высокая скидка (>10%): 0 или 1")
    month_num: int = Field(6, description="Номер месяца (1-12)")
    quarter: str = Field("Q2", description="Квартал (Q1/Q2/Q3/Q4)")
    month: str = Field("June", description="Название месяца")
    country: str = Field("USA", description="Страна")
    region: str = Field("North America", description="Регион")
    city: str = Field("New York", description="Город")
    product_name: str = Field("Samsung Galaxy S23", description="Название продукта")
    category: str = Field("Smartphone", description="Категория товара")
    storage: str = Field("128GB", description="Объём памяти")
    color: str = Field("Phantom Black", description="Цвет")
    is_5g: str = Field("Yes", description="Поддержка 5G (Yes/No)")
    currency: str = Field("USD", description="Валюта")
    sales_channel: str = Field("E-commerce Platform", description="Канал продаж")
    payment_method: str = Field("Credit Card", description="Способ оплаты")
    customer_segment: str = Field("Individual", description="Сегмент покупателя")
    customer_age_group: str = Field("25-34", description="Возрастная группа покупателя")
    previous_device_os: str = Field("Android (Samsung)", description="Предыдущая ОС устройства")


class PredictionOutput(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "model": "CatBoost", "classes": list(label_encoder.classes_)}


@app.post("/predict", response_model=PredictionOutput)
def predict(sale: SaleInput) -> PredictionOutput:
    try:
        row = pd.DataFrame([sale.model_dump()])
        for col in numeric_features:
            row[col] = pd.to_numeric(row[col], errors="coerce")

        x_proc = preprocessor.transform(row)
        pred_enc = int(np.array(model.predict(x_proc)).flatten()[0])
        proba = np.array(model.predict_proba(x_proc)).flatten()

        predicted_class = label_encoder.inverse_transform([int(pred_enc)])[0]
        proba_dict = {cls: float(p) for cls, p in zip(label_encoder.classes_, proba)}

        return PredictionOutput(predicted_class=predicted_class, probabilities=proba_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict_batch")
def predict_batch(sales: list[SaleInput]) -> list[PredictionOutput]:
    return [predict(sale) for sale in sales]
