import zipfile
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error



# 1. Загрузка данных


zip_path = r'F:\Data_ML\house-prices-advanced-regression-techniques.zip'

with zipfile.ZipFile(zip_path) as z:
    train_df = pd.read_csv(z.open('train.csv'))
    test_df = pd.read_csv(z.open('test.csv'))

# сохраняем Id для submission
test_ids = test_df['Id']

# удаляем Id из признаков
train_df = train_df.drop('Id', axis=1)
test_df = test_df.drop('Id', axis=1)



# 2. Разделение X / y


X = train_df.drop('SalePrice', axis=1)

# лог-трансформация target (КРИТИЧЕСКИ ВАЖНО)
y = np.log1p(train_df['SalePrice'])



# 3. Preprocessing


num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer,
     make_column_selector(dtype_include=['int64', 'float64'])),
    ('cat', cat_transformer,
     make_column_selector(dtype_include='object'))
])



# 4. Pipeline (preprocess + model)


pipe = Pipeline([
    ('preprocess', preprocessor),
    ('model', Ridge(alpha=1.0))
])



# 5. Валидация


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe.fit(X_train, y_train)

val_preds = pipe.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, val_preds))
print(f'Validation RMSE: {rmse:.4f}')



# 6. Кросс-валидация


cv_scores = cross_val_score(
    pipe,
    X,
    y,
    scoring='neg_root_mean_squared_error',
    cv=5
)

print(f'CV RMSE: {-cv_scores.mean():.4f}')



# 7. Финальное обучение + prediction


pipe.fit(X, y)

test_preds = np.expm1(pipe.predict(test_df))


# 8. Submission


submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_preds
})

submission.to_csv('submission.csv', index=False)
print('submission.csv сохранён')
