import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('train_dataset.csv')
print(f'Форма тренировочных данных: {train_data.shape}')

train_data['datetime'] = pd.to_datetime(train_data['METEOFORECASTHOUR_OPENM_Datetime'])
train_data['year'] =        train_data['datetime'].dt.year
train_data['day'] =         train_data['datetime'].dt.day
train_data['day_of_week'] = train_data['datetime'].dt.dayofweek
train_data['day_of_year'] = train_data['datetime'].dt.dayofyear

train_data['month_sin'] = np.sin(2 * np.pi * train_data['datetime'].dt.month / 12)
train_data['month_cos'] = np.cos(2 * np.pi * train_data['datetime'].dt.month / 12)
train_data['hour_sin'] =  np.sin(2 * np.pi * train_data['datetime'].dt.hour / 24)
train_data['hour_cos'] =  np.cos(2 * np.pi * train_data['datetime'].dt.hour / 24)

train_data = train_data.sort_values('datetime')

train_data = train_data.drop(columns=['METEOFORECASTHOUR_OPENM_Datetime','datetime'])

#удалим недействительные данные

train_data = train_data.dropna()
print(f'Форма тренировочных данных после очистки недействительных данных: {train_data.shape}')

#определяем целевую переменную и признаки

target = 'Выработка. Результирующий расчет'
features = [col for col in train_data.columns if col != target]

x_train = train_data[features]
y_train = train_data[target]

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.15, random_state = 42, shuffle = False)

model = CatBoostRegressor(
    iterations = 3000,
    learning_rate = 0.03,
    depth = 5,
    loss_function = "RMSE",
    
    l2_leaf_reg = 5,
    random_strength = 1.5,
    bagging_temperature = 0.8,

    early_stopping_rounds = 200,
    eval_metric = "RMSE",
    random_seed = 42,
    verbose = 300
    )

model.fit(x_train,y_train, eval_set = (x_val, y_val))
print('обучение завершено')

valid_data = pd.read_csv('valid_features.csv')

print(f'Форма валидационных данных: {valid_data.shape}')

valid_data['datetime'] = pd.to_datetime(valid_data['METEOFORECASTHOUR_OPENM_Datetime'])
valid_data['year'] =        valid_data['datetime'].dt.year
valid_data['day'] =         valid_data['datetime'].dt.day
valid_data['day_of_week'] = valid_data['datetime'].dt.dayofweek
valid_data['day_of_year'] = valid_data['datetime'].dt.dayofyear

valid_data['month_sin'] = np.sin(2 * np.pi * valid_data['datetime'].dt.month / 12)
valid_data['month_cos'] = np.cos(2 * np.pi * valid_data['datetime'].dt.month / 12)
valid_data['hour_sin'] =  np.sin(2 * np.pi * valid_data['datetime'].dt.hour / 24)
valid_data['hour_cos'] =  np.cos(2 * np.pi * valid_data['datetime'].dt.hour / 24)

valid_data = valid_data.drop(columns=['METEOFORECASTHOUR_OPENM_Datetime','datetime'])

missing_features = set(x_train.columns) - set(valid_data.columns)

if missing_features:
    print(f'В валидационных даннах отсутствуют признаки: {missing_features}')

x_valid = valid_data[x_train.columns]

print('Прогнозируем...')

predictions = model.predict(x_valid)
print(f'Количество предсказаний: {len(predictions)}')

predictions = np.clip(predictions, 0, None)

submission = pd.DataFrame({'target':predictions})
submission.to_csv('submission.csv', index = False, header = True)