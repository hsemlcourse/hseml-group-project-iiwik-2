import requests
import streamlit as st

API_URL = "http://api:8000/predict"

st.set_page_config(
    page_title="Samsung Return Predictor",
    page_icon="📱",
    layout="wide",
)

st.title("📱 Samsung Return Status Predictor")
st.markdown(
    "Предсказывает, что покупатель сделает с товаром: **оставит**, **вернёт** или **обменяет**."
)

st.sidebar.header("Параметры продажи")

with st.sidebar:
    st.subheader("📦 Товар")
    product_name = st.selectbox(
        "Название продукта",
        ["Samsung Galaxy S23", "Samsung Galaxy A54", "Samsung Galaxy Z Fold5",
         "Samsung Galaxy Tab S9", "Samsung Galaxy Watch6"],
    )
    category = st.selectbox("Категория", ["Smartphone", "Tablet", "Wearable", "Laptop"])
    storage = st.selectbox("Память", ["64GB", "128GB", "256GB", "512GB", "1TB"])
    color = st.selectbox("Цвет", ["Phantom Black", "Cream", "Green", "Lavender", "Graphite"])
    is_5g = st.selectbox("5G", ["Yes", "No"])

    st.subheader("💰 Цена и скидка")
    unit_price = st.number_input(
    "Цена (USD)",        min_value=50.0, max_value=3000.0, value=999.0, step=50.0
)
    discount_pct = st.slider("Скидка (%)", 0, 50, 5)
    units_sold = st.number_input("Количество", min_value=1, max_value=100, value=1)
    discounted_price = unit_price * (1 - discount_pct / 100)
    revenue = discounted_price * units_sold
    discount_amount = unit_price * discount_pct / 100
    is_high_discount = 1 if discount_pct > 10 else 0

    st.subheader("🌍 География и канал")
    country = st.selectbox("Страна", ["USA", "Germany", "India", "Brazil", "Japan", "UK", "France"])
    region_map = {
        "USA": "North America", "Germany": "Europe", "India": "Asia Pacific",
        "Brazil": "Latin America", "Japan": "Asia Pacific", "UK": "Europe", "France": "Europe",
    }
    region = region_map.get(country, "Other")
    city = st.text_input("Город", "New York")
    sales_channel = st.selectbox(
        "Канал продаж",
        ["E-commerce Platform", "Authorized Reseller", "Corporate / B2B",
         "Third-Party Retailer", "Samsung Store"],
    )
    payment_method = st.selectbox(
        "Способ оплаты",
        ["Credit Card", "Debit Card", "Samsung Pay", "Net Banking",
         "Gift Card", "BNPL (Buy Now Pay Later)"],
    )
    currency_map = {"USA": "USD", "Germany": "EUR", "India": "INR", "Brazil": "BRL",
                    "Japan": "JPY", "UK": "GBP", "France": "EUR"}
    currency = currency_map.get(country, "USD")
    fx_map = {"USD": 1.0, "EUR": 0.92, "INR": 83.0, "BRL": 5.0, "JPY": 149.0, "GBP": 0.79}
    fx_rate = fx_map.get(currency, 1.0)
    revenue_local = revenue * fx_rate

    st.subheader("👤 Покупатель")
    customer_segment = st.selectbox(
    "Сегмент", ["Individual", "Business", "Enterprise", "Government"]
)
    customer_age_group = st.selectbox("Возраст", ["18-24", "25-34", "35-44", "45-54", "55+"])
    previous_os = st.selectbox(
        "Предыдущая ОС",
        ["Android (Samsung)", "Android (Other)", "iOS", "New User", "Feature Phone"],
    )
    customer_rating = st.slider("Рейтинг покупателя", 1.0, 5.0, 4.0, 0.1)

    st.subheader("Дата")
    year = st.selectbox("Год", [2022, 2023, 2024])
    month_num = st.slider("Месяц", 1, 12, 6)
    quarter_map = {1: "Q1", 2: "Q1", 3: "Q1", 4: "Q2", 5: "Q2", 6: "Q2",
                   7: "Q3", 8: "Q3", 9: "Q3", 10: "Q4", 11: "Q4", 12: "Q4"}
    month_name_map = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May",
                      6: "June", 7: "July", 8: "August", 9: "September",
                      10: "October", 11: "November", 12: "December"}
    quarter = quarter_map[month_num]
    month_name = month_name_map[month_num]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Цена со скидкой", f"${discounted_price:.2f}")
with col2:
    st.metric("Выручка (USD)", f"${revenue:.2f}")
with col3:
    st.metric("Сумма скидки", f"${discount_amount:.2f}")

st.markdown("---")

if st.button("🔮 Предсказать статус возврата", type="primary", use_container_width=True):
    payload = {
        "year": year,
        "unit_price_usd": unit_price,
        "discount_pct": discount_pct,
        "units_sold": units_sold,
        "discounted_price_usd": discounted_price,
        "revenue_usd": revenue,
        "fx_rate_to_usd": fx_rate,
        "revenue_local_currency": revenue_local,
        "customer_rating": customer_rating,
        "discount_amount": discount_amount,
        "revenue_per_unit": revenue / units_sold,
        "is_high_discount": is_high_discount,
        "month_num": month_num,
        "quarter": quarter,
        "month": month_name,
        "country": country,
        "region": region,
        "city": city,
        "product_name": product_name,
        "category": category,
        "storage": storage,
        "color": color,
        "is_5g": is_5g,
        "currency": currency,
        "sales_channel": sales_channel,
        "payment_method": payment_method,
        "customer_segment": customer_segment,
        "customer_age_group": customer_age_group,
        "previous_device_os": previous_os,
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()

        pred = result["predicted_class"]
        proba = result["probabilities"]

        emoji_map = {"Kept": "✅", "Returned": "🔄", "Exchanged": "🔁"}
        color_map = {"Kept": "green", "Returned": "red", "Exchanged": "orange"}

        st.markdown(f"## Результат: {emoji_map.get(pred, '')} **:{color_map[pred]}[{pred}]**")

        st.subheader("Вероятности по классам")
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        for i, (cls, prob) in enumerate(sorted(proba.items(), key=lambda x: -x[1])):
            cols[i].metric(
                f"{emoji_map.get(cls, '')} {cls}",
                f"{prob * 100:.1f}%",
            )
            cols[i].progress(prob)

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Не удалось подключиться к API. "
            "Запустите FastAPI сервер: `uvicorn app.main:app --reload`")
    except Exception as e:
        st.error(f"Ошибка: {e}")

st.markdown("---")
st.caption("Модель: CatBoost | Метрика: F1-macro = 0.333 | Данные: Samsung Global Sales Dataset")
