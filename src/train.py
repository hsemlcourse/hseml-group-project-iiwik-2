import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

df = pd.read_csv('data/raw/samsung_global_sales_dataset.csv')
df['sale_date'] = pd.to_datetime(df['sale_date'])
df['discount_amount'] = df['unit_price_usd'] * df['discount_pct'] / 100
df['revenue_per_unit'] = df['revenue_usd'] / df['units_sold']
df['is_high_discount'] = (df['discount_pct'] > 10).astype(int)
df['month_num'] = df['sale_date'].dt.month

X = df.drop(columns=['return_status', 'sale_id', 'sale_date'])
y = df['return_status']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
)

cat_features = X_train.select_dtypes(include=['object']).columns.tolist()
num_features = X_train.select_dtypes(include=np.number).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
])
X_train_proc = preprocessor.fit_transform(X_train)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

model = CatBoostClassifier(
    auto_class_weights='Balanced',
    random_seed=RANDOM_STATE,
    verbose=100,
)
model.fit(X_train_proc, y_train_enc)

Path('models').mkdir(exist_ok=True)
with open('models/catboost_pipeline.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'preprocessor': preprocessor,
        'label_encoder': le,
        'numeric_features': num_features,
        'categorical_features': cat_features,
    }, f)

print('Модель сохранена в models/catboost_pipeline.pkl')