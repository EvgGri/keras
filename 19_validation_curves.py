# Проверочные кривые - это полезный инструмент для улучшения качества модели благодаря решению таких проблем, как переобучение или
# недообучение. Проверочные кривые связаны с кривыми обучения, но только вместо того, чтобы строить рафик верностей на тренировочном и тестовом
# наборах как функциях от объема образца, мы варьируем значения модельных параметров, например параметр обратной регуляризации С в логистической
# регрессии.

# -=-=-=-=-=-=-=-=-=-=-

# Способы применения кривых обучения для диагностирования у алгоритма обучения проблемы с переобучением (высокой дисперсией) или недообучением
# (высоким смещением).

# Если модель для данного тренировочного набора данных слишком сложна, например, в этой модели имеется слишком много степеней свободы либо
# параметров, то она демонстрирует тенденцию к переобучению и плохому обобщению на новых, невстречаемых ранее данных.

# Построив график модельных верностей на тренировочном и проверочном наборах как функций размера тренировочного набора, можно легко установить,
# страдает ли модель от высокой дисперсии или высокого смещения и поможет ли аккумулирование большого количества данных решить эту проблему.

# Если модель имеет низкую верность на тренировочном, так и на перекрестно-проверочном наборе данных, что указывает на то, что она недообучена
# под тренировочные данные, то эту проблему можно решить, за счет увеличения числа параметров модели, например путем аккумулирования или создания
# дополнительных признаков или уменьшения степени регуляризации, например, в классификаторах на методах линейной регрессии или опорных векторов.

# Если модель демонстрирует хорошие результаты на тренировочном наборе данных и плохие результаты на перекрестно-проверочном наборе данных, т.е.
# у модели большая дисперсия, то можно постараться устранить данный недостаток с помощью аккумулирования большего количества тренировочных данных
# или либо уменьшить сложность модели (число входящих параметров), например, увеличением параметра регуляризации. В случае нерегуляризованных
# моделях может помочь сокращение отбора признаков методами отбора признаков, либо выделения признаков.


import pandas as pd
import numpy as np
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
# df = pd.read_csv(url, header=None)

url = '/Users/grigorev-ee/Work/AnacondaProjects/My_projects/ml_w_python/data/cancer.base'
df = pd.read_csv(url)


names = ['id_number', 'diagnosis', 'radius_mean',
         'texture_mean', 'perimeter_mean', 'area_mean',
         'smoothness_mean', 'compactness_mean', 'concavity_mean',
         'concave_points_mean', 'symmetry_mean',
         'fractal_dimension_mean', 'radius_se', 'texture_se',
         'perimeter_se', 'area_se', 'smoothness_se',
         'compactness_se', 'concavity_se', 'concave_points_se',
         'symmetry_se', 'fractal_dimension_se',
         'radius_worst', 'texture_worst', 'perimeter_worst',
         'area_worst', 'smoothness_worst',
         'compactness_worst', 'concavity_worst',
         'concave_points_worst', 'symmetry_worst',
         'fractal_dimension_worst']

df = df[names]

# Присвоим 30 признакоы массиву X библиотеки NumPy.
# При помощи LabelEncoder преобразуем метки классов из их исходного строкового представления (M и B) в целочисленное.
from sklearn.preprocessing import LabelEncoder
X = df.iloc[:,2:].values
y = df.iloc[:,1].values

print(y[:36])

le = LabelEncoder()
y=le.fit_transform(y)

# M-злокачественная опухоль (malignant), B-доброкачественная (benign). Выведем первые 36 элементов из набора данных.
print(y[:36])

# Выведем пример того, как работает кодировщик на тестовом наборе данных
le.transform(['M','M','B'])

# Прежде, чем в следующем подразделе мы построим наш первый модельный конвейер, рзделим выборку на обучающую и тренировочную.
# Оценка модели на ранее не встречавшихся данных
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Совмещение преобразователей и оценщиков в конвейере
print(X_train[:,1])

# Вместо раздельного выполнения шагов по подгонке и преобразованию тренировочного и тестового наборов данных мы можем расположить бъекты
# StandardScaler, PCA и LogisticRegression друг за другом в конвейере.
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# -=-=-=-=-=-=-= Основной код после подготовки

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression(penalty='l2', random_state=0))])

# Аналогично функции learning_curve, функциф validation_curve для оценки работоспособности алгоритма по умолчанию использует стратифицированную
# k-блочную перекрестную проверку, в случае если мы используем алгоритмы классификации данных. Внутри функции validation_curve мы указали параметр,
# который мы хотели оценить. В данном случае, это параметр обратной регуляризации С классификатора LogisticRegression. Мы его обзначили, как
# clf__C, чтобы иметь доступ к объекту LogisticRegression внутри конвейера библиотеки scikit-learn для заданного диапазона значений, который
# мы задала посредством параметра param_range. Аналогично примеру кривой обучения в предыдущем разделе мы построим график средних верностей на
# тренировочных данных и перекрестно-проверочных данных и соответствующих им стандартных отклонений.
from sklearn.model_selection import validation_curve
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(estimator = pipe_lr, X=X_train, y=y_train, param_name='clf__C', param_range=param_range, cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='тренировочная верность')
plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='проверочная верность')
plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Параметр С')
plt.ylabel('Верность')
plt.ylim([0.8, 1.0])
plt.show()

# Несмотря на то, что различия в верности для варьирующихся значений парамиетра С незначительны, мы видим, что при увеличении силы регуляризации,
# модель слегка недоадаптируется под данные.
