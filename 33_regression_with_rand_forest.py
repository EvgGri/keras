# -=-=-=-=-=-=-=-=-=- Обработка нелинейных связей при помощи случайных лесов
# Случайный лес - ансамбль из двух и более деревьев решений, это сумма кусочно-линейных функций.
# Другими словами, мы подразделяем входное пространство на на области меньшего размера, которые становятся более управляемыми.

# Нам не требуется преобразование признаков при использовании алгоритма дерева решений, если мы имеем дело с нелинейностью.
# Мы выращиваем дерево решений путем итеративного расщепления, пока не будут выполнены необходимые критерии остановки.
# В задаче классификации мы максимизировали прирост информации (Information Gain, IG), а энтропию определили, как меру неоднородности.
# Мы хотим найти расщепление признака, которое сокращает неоднородности в дочерних узлах.

# При использовании дерева решений для регрессии мы заменим энтропию на MSE.
# В контексте регрессии на основе дерева решений MSE часто упоминается, как внутриузловая дисперсия.

# Набор данных, содержащий информацию о зданиях в пригороде Бостона.
import pandas as pd
url='./data/BostonHousing.xls'

df=pd.read_excel(url, index_col=None)
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df.head()


from sklearn.tree import DecisionTreeRegressor

X = df[['LSTAT']].values
y = df['MEDV'].values

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)
sort_idx = X.flatten().argsort()

# Реализация обычной регрессионной модели методом наименьших квадратов (OLS - Ordinary least squares)
from sklearn.linear_model import LinearRegression
slr = LinearRegression()


slr.fit(X,y)
print('Наклон: %.3f' % slr.coef_[0])
print('Пересечение: %.3f' % slr.intercept_)

def lin_regplot(X,y, model):
    plt.scatter(X,y,c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None

import matplotlib.pyplot as plt
lin_regplot(X,y,slr)

lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('log(% населения с более низким статусом [LSTAT])')
plt.ylabel('Цена в тыс. долл. [MEDV]')
plt.show()
# -=-=-=-=-=-=-=-=-=- Регрессия на основе случайного леса
# Случайный лес - ансамблевый метод, который объединяет 2 и более деревьев решений.
# Как правило, единственный настроечный параметр в случаном лесу - это количество деревьев.
# Базовый алгоритм случайного леса для регрессии практически идентичен алгоритму случайного леса для классификации.
# Разница состоит в том, что для роста отдельных деревьев решений мы используем критерий MSE и предсказанная целевая переменная
# вычисляется как усредненное предсказание по всем деревьям решений.


# Набор данных, содержащий информацию о зданиях в пригороде Бостона.
import pandas as pd
url='./data/BostonHousing.xls'

df=pd.read_excel(url, index_col=None)
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df.head()


# Воспользуемся всеми признаками из набора данныз BostonHousing
X = df.iloc[:, :-1].values
y = df['MEDV'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=1000,
                               criterion="mse",
                               random_state=1,
                               n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print('\nMSE тренировка: %.3f, тестирование: %.3f' % (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))
print('\nR^2 тренировка: %.3f, тестирование: %.3f' % (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))
import matplotlib.pyplot as plt
# Случайный лес демонстрируем переподгонку под тренировочные данные.
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Тренировочные данные')
plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Тестовые данные')
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10,50])
plt.show()
 # Видно, что модель работает гораздо лучше соответствует тренировочным данным, чем тестовым, т.к. мы видим выбросы вдоль оси У.
 # Это показывает коэффициент детерминации R^2.
