# Набор данных, содержащий информацию о зданиях в пригороде Бостона.
import pandas as pd
url='./data/BostonHousing.xls'

df=pd.read_excel(url, index_col=None)
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df.head()

# Смоделируем связь между ценами на жилье и LSTAT (процентом населения с более низким статусом) с использованием полинома второй степени
# и третье степени и сравним его с линейной подгонкой
from sklearn.preprocessing import PolynomialFeatures
X=df[['LSTAT']].values
y=df[['MEDV']].values

from sklearn.linear_model import LinearRegression
regr = LinearRegression()

# Создаем полиномиальные признаки
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)

X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# Линейная подгонка
import numpy as np
from sklearn.metrics import r2_score
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

# Квадратичная подгонка
regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

# Кубическая подгонка
regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

# Строим график с результатами
import matplotlib.pyplot as plt
plt.scatter(X, y, label='Тренировочеые точки', color='lightgray')
plt.plot(X_fit, y_lin_fit, label='линейная (d=1), $R^2=%.2f$' % linear_r2, color='blue', lw=2, linestyle=':')
plt.plot(X_fit, y_quad_fit, label='квадратичная (d=2), $R^2=%.2f$' % quadratic_r2, color='red', lw=2, linestyle='-')
plt.plot(X_fit, y_cubic_fit, label='кубическая (d=3), $R^2=%.2f$' % cubic_r2, color='green', lw=2, linestyle='--')
plt.xlabel('% населения с более низким статусом [LSTAT]')
plt.ylabel('Цена в тыс. долл. [MEDV]')
plt.legend(loc='upper right')
plt.show()
# Как видно на получившемся графике, кубическая подгонка захватывает связь перемнных лучше, чем линейная и квадратичная.
# Однако мы должны понимать, что добавления все большего числа полиномиальных признаков увеличивает сложность модели и поэтому
# увеличивает шанс ее переобучения
# Поэтому на практике рекомендуется проверять работу модели на отдельном тестовом наборе данных.

# -=-=-=-=-=-=-=-=- Использование полиномиальных связей не всегда является наилучшим решением.
# В данном примере можно было легко использовать функции преобразования (логарифмическое и квадратный корень), которые перевели бы
# исходное пространство признаков в линейное пространство.

# Преобразование признаков
X_log = np.log(X)
y_sqrt = np.sqrt(y)

# Выполнить подгонку признаков
X_fit = np.arange(X_log.min()-1, X_log.max()+1,1)[:,np.newaxis]
regr = regr.fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

# Построим график с результатами
plt.scatter(X_log, y_sqrt, label='Тренировочеые точки', color='lightgray')
plt.plot(X_fit, y_lin_fit, label='линейная (d=1), $R^2=%.2f$' % linear_r2, color='blue', lw=2, linestyle=':')
plt.xlabel('log(% населения с более низким статусом [LSTAT])')
plt.ylabel('$\sqrt{Цена в тыс. долл. [MEDV]}$')
plt.legend(loc='lower left')
plt.show()
