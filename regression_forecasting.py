import pandas as pd
import quandl, datetime, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

#импортирование данных

quandl.ApiConfig.api_key = 'QnBdjmbzJ7NxzD-FKSth'
df = quandl.get_table('WIKI/PRICES')

#формирование таблицы

df.index = df['date']
df = df.drop(['date'], 1)
df = df.sort_index(axis=0, ascending=True)
df = df[['adj_open','adj_high','adj_low','adj_close','adj_volume']]

df['HL_PCT'] = (df['adj_high'] - df['adj_close']) / df['adj_close'] * 100.0
df['PCT_change'] = (df['adj_close'] - df['adj_open']) / df['adj_open'] * 100.0
df = df[['adj_close','HL_PCT','PCT_change','adj_volume']]
df.fillna(-99999, inplace=True)

# Целью модели является прогноз цены закрытия торгов по акции компании на 10 дней (одна тысячная размера наблюдений)
# вперед после закрытия торгов, поэтому создается еще одна колонка 'forecast_col',
# которая содержит в себе значения эндогенной переменной

forecast_col = 'adj_close'

# Далее значения 'y' сдвигаются вверх на одну тысячну размера выборки (10 денй)
# Таким образом, каждому наблюдению иксов будет соответствовать цена акции на момент
# закрытия торгов через 10 дней
forecast_out = int(math.ceil(0.001*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

# создания массива иксов
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X) # вещь, которую я так и не понял
X_before = X[:-forecast_out] # отдельный массив для иксов, для которых есть соответсвтующий 'y'
X_after = X[-forecast_out:] # отдельный массив без 'y', пока

df.dropna(inplace = True) # удаление из таблицы всех строк, где есть нечисловые
                          # или пустые значения

y = np.array(df['label']) # создание массива 'y'

# тестирование модели

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_before, y, test_size=0.2)

clf = LinearRegression(n_jobs = -1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)

# прогнозирование

forecast_set = clf.predict(X_after)


df['forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# создание новых строк в таблице со спрогнозированными переменными
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

# визуализация

df['adj_close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

