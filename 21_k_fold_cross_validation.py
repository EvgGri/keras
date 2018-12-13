# Отбор алгоритмов методов вложенной перекрестной проверки
# Использование к-блочной перекрестной проверки  сочетании с оптимизационным поиском по сетке параметров - хороший подход.
# Если мы хотим выполнить отбор среди разных алгоритмов, рекомендуется использовать вложенную перекрестную проверку.

# Во вложенной перекрестной проверке имеется внешний цикл к-блочной перекрестной проверки для разделения данных на тренировочные и
# тестовые блоки и внутренний цикл, который используется для отбора модели при помощи к-блочной перекрестной проверки на тренировочном
# блоке. После отбора модели затем тестовый блок используется для оценки качества модели.

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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn. model_selection import GridSearchCV
from sklearn.svm import SVC
pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range,
               'clf__kernel': ['linear']},
              {'clf__C': param_range,
               'clf__gamma': param_range,
               'clf__kernel': ['rbf']}]

# -=-=-=-=-=-=-= Основной алгоритм -=-=-=-=-=-=-=-=-=-= перекрестная проверка типа 5x2

# Инициализируем объект GridSearchCV, чтобы натренировать и настроить конвейер методо опорных векторов (SVM).
# Мы назначаем параметру param_grid объекта GridSearchCV список словарей с определением параметров, которые мы хотели бы настроить.
# Для линейного классификатора SVM мы настраиваем только один параметр, параметр обратной регуляризации C, для ядерного SVM c РБФ в
# качестве ядра мы выполнили настройку двух параметров: C и gamma. Отметим, что параметр gamma имеет непосредственное отношение к
# ядерным методам SVM. По результатам применения оптимизационного поиска по сетке параметров в трибуте best_score мы получим оценку
# наиболее качественной модели и обратились к его параметрам через атрибут best_params.

from sklearn.model_selection import cross_val_score
gs=GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=2, n_jobs=-1)
scores=cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('Перекрестно-проверочная верность: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# Результирующая средняя верность на перекрестно-проверочных данных дает хорошую оценку того, чего ожидать, если настроить гиперпараметры
# модели и затем использовать ее на ранее не встречавшихся данных. Например, мы можем применить подход с вложенной перекрестной проверкой
# для сравнения модели на основе SVM с классификатором на основе простого дерева решений, для простоты мы настроим лишь параметр ее глубины.
from sklearn.tree import DecisionTreeClassifier
gs=GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), param_grid=[{'max_depth': [1,2,3,4,5,6,7,None]}],
                scoring='accuracy', cv=5, n_jobs=-1)
scores=cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('Перекрестно-проверочная верность: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# Как видно, качество работы на данных вложенной перекрестной проверки модели SVM лучше качества модели дерева решений.
# Поэтому они ожидаемо может быть более подходящим выбором для классификации новых данных.
