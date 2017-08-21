import pandas as pd
import quandl
import numpy as np
import math
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

# импортирование данных

quandl.ApiConfig.api_key = 'QnBdjmbzJ7NxzD-FKSth'
df = quandl.get_table('WIKI/PRICES')

# редактирование таблицы

df = df[['adj_open','adj_high','adj_low','adj_close','adj_volume']]

# создание новых столбцов

df['HL_PCT'] = (df['adj_high'] - df['adj_close']) / df['adj_close'] * 100.0
df['PCT_change'] = (df['adj_close'] - df['adj_open']) / df['adj_open'] * 100.0
df = df[['adj_close','HL_PCT','PCT_change','adj_volume']]

# Модель должна прогнозировать цену акции компании на момент закрытия торгов
# через 10 дней после исследуемого периода. Поэтому колонка эндогенной
# переменной в таблице представляет собой сдвинутую вверх колонку цен акций
# на момент закрытия торгов на 1 тысячную размера выборки (10 дней)

forecast_col = 'adj_close'
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)

# создание массивов эндогенных и экзогенных переменных

X = np.array(df.drop(['label'],1))
y = np.array(df['label'])

X = preprocessing.scale(X)

# тестирование моделей

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(k,confidence)


