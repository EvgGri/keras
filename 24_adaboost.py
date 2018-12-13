# AdaBoost - адаптивный бустинг (adaptive boosting)

# В бустинге ансамбль состоит из простых классификаторов или слабых классификаторов, имеющих незначительно преимущество в качестве над
# случайным гаданием. Типичным примером слабого классификатора является одноуровневое дерево решений (пенек решения).
# Лежащая в основе бустинга ключевая идея заключается в том, что он сосредоточен на тренировочных образцах, которые трудно классифицировать,
# т.е. с целью улучшения качества ансамбля бустинг дает слабым ученикам в последствии обучиться на ошибочно классифицированных тренировочных
# образцах.

# В отличие от бэггинга, алгоритм бустинга использует случайные подможества тренировочных образцов, извлеченные из тренировочного набора данных
# без возврата. Таким образом алгоритм бустинга можно записать в 4-х основных шагах:

# 1. извлечь случайное подмножество тренировочных образцов d_1 без возврата из тренировочного набора D для тренировки слабого ученика С_1
# 2. извлечь второе случайное тренировочное подмножество d_2 без возврата из тренировочного набора и добавить 50% ранее ошибочно
# классифицированных образцов для тренировки слабого ученика С_2
# 3. найти в тренировочном наборе D тренировочные образцы d_3, по которым C_1 & C_2 расходятся, для тренировки третьего слабого ученика С_3
# 4. Объединить слабых учеников С_1, С_2, С_3 посредством мажоритарного голосования

# В отличие от описаной здесь процедуры бустинга, алгоритм AdaBoost для тренировки слабых учеников использует полный тренировочный набор, где
# тренировочные образцы взвешиваются повторно в каждой итерации с целью построения сильного классификатора, который обучается на ошибках
# предыдущих слабых классификаторов в ансамбле.

# Разбивка на тренировочное и тестовое подмножество

import pandas as pd
import numpy as np

from sklearn.datasets import load_wine
data = load_wine()
list(data.target_names)

# url='https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
# df_wine=pd.read_csv(url, header=None)

df_wine = pd.DataFrame(data.data)

df_wine.columns=['Алкоголь','Яблочная кислота','Зола',
                 'Щелочность золы','Магний','Всего фенола','Флаваноиды',
                 'Фенолы нефлаваноидные','Проантоцианины','Интенсивность цвета',
                 'Оттенок','OD280 / OD315 разбавленных вин','Пролин']

df_wine['Метка класса'] = pd.DataFrame(data.target)

# Выберем только классы 2 и 3
df_wine = df_wine[df_wine['Метка класса'] != 0]

# Разбиение на train и test выборки
# X, y = df_wine[['Алкоголь','Яблочная кислота','Зола','Щелочность золы','Магний','Всего фенола','Флаваноиды','Фенолы нефлаваноидные',
#                 'Проантоцианины','Интенсивность цвета','Оттенок','OD280 / OD315 разбавленных вин','Пролин']].values,df_wine['Метка класса'].values

X, y = df_wine[['Алкоголь', 'Оттенок']].values, df_wine['Метка класса'].values

# Приведем метки классов в двоичный формат
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


# В sikit-learn алгоритм уже реализован

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=0)
ada = AdaBoostClassifier(base_estimator=tree, n_estimators=500, learning_rate=0.1, random_state=0)

from sklearn.metrics import accuracy_score
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)

print('Верность дерева решений на тренировочном/тестовом наборах %.3f/%.3f' % (tree_train, tree_test))

# Как видно, пенек дерева решений показывает тенденцию к недообучению под тренировочные данные, в отличие от неподрезанного дерева решений

ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)

ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)

print('Верность дерева решений на тренировочном/тестовом наборах %.3f/%.3f' % (ada_train, ada_test))
# Как видно, модель AdaBoost правильно идентифицирует все метки классов тренировочного набора и также показывает слегка улучшенное качество
# на тестовом наборе, по сравнению с пеньком дерева решения. Однако мы видим, что вместе с нашей попыткой уменьшить смещение, мы привнесли
# дополнительную дисперсию.

# Визуализация на первых двух компонентах
import matplotlib.pyplot as plt

x_min = X_train[:,0].min()-1
x_max = X_train[:,0].max()+1
y_min = X_train[:,1].min()-1
y_max = X_train[:,1].max()+1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(18,7))

for idx, clf, tt in zip([0,1], [tree, ada], ["Дерево решений","АдаБуст"]):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='blue', marker='^')
    axarr[idx].scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='red', marker='o')
    axarr[idx].set_title(tt)

axarr[0].set_ylabel('Главная компонента №1', fontsize=12)
# axarr[1].set_ylabel('Главная компонента №2', fontsize=12)
plt.text(10, -1, s='Главная компонента №2', ha='center', va='center', fontsize=12)
plt.show()
# В итоге, зачастую, используя ансамблевые методы, мы получаем скромный прирост в точности, но значимый прирост в сложности вычислений. 
