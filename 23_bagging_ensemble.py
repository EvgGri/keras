# Bagging = bootstrap aggregating, агрегирование бутстрап-выборок
# Вместо того, чтобы для подгонки ансамблевых классификаторов использовать один и тот же тренировочный набор данных,
# мы используем бутстрап-выборки (случайные образцы с возвратом).
# Каждая бутстрап-выборка затем используется для подгонки классификатора C_jself.

# Разбивка на тренировочное и тестовое подмножество

import pandas as pd
import numpy as np

from sklearn.datasets import load_wine
data = load_wine()
data.target[[10, 80, 140]]
list(data.target_names)

# url='https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
# df_wine=pd.read_csv(url, header=None)

df_wine = pd.DataFrame(data.data)

df_wine.columns=['Алкоголь','Яблочная кислота','Зола',
                 'Щелочность золы','Магний','Всего фенола','Флаваноиды',
                 'Фенолы нефлаваноидные','Проантоцианины','Интенсивность цвета',
                 'Оттенок','OD280 / OD315 разбавленных вин','Пролин']
df_wine.head()
df_wine['Метка класса'] = pd.DataFrame(data.target)

# Выберем только классы 2 и 3
df_wine = df_wine[df_wine['Метка класса'] != 0]

df_wine.describe()
# Размер data frame'a
df_wine.shape

# Разбиение на train и test выборки
X, y = df_wine[['Алкоголь', 'Оттенок']].values, df_wine['Метка класса'].values

# Приведем метки классов в двоичный формат
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


# В sikit-learn алгоритм уже реализован алгоритм бэггинг-классификатора BaggingClassifier. В данном примере в качестве базового
# классификатора мы будем использовать  неподрезанное дерево решений и создадим ансамбль из 500 деревьев решений, подогнанных на
# разных бутстрап-выборках из тренировочного набора.

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=1)
bag = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples = 1.0, max_features=1.0,
                        bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=1)

from sklearn.metrics import accuracy_score
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)

print('Верность дерева решений на тренировочном/тестовом наборах %.3f/%.3f' % (tree_train, tree_test))

# Модель хорошо работает на тренировочном наборе данных, но на тестовом точность значительно ниже, что свидетельствует о переобучении.

bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)

bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)

print('Верность дерева решений на тренировочном/тестовом наборах %.3f/%.3f' % (bag_train, bag_test))

# Визуализация на первых двух компонентах
import matplotlib.pyplot as plt

x_min = X_train[:,0].min()-1
x_max = X_train[:,0].max()+1
y_min = X_train[:,1].min()-1
y_max = X_train[:,1].max()+1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(18,7))

for idx, clf, tt in zip([0,1], [tree, bag], ["Дерево решений","Бэггинг"]):
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

# На практике более сложные задачи классификации и высокая размерность наборов данных могут легко привести к переобучению в одиночных
# деревьях решений, и именно здесь алгоритм бэггинга может проявить свою силу. Бэггинг - эффективный способ снижения дисперсии модели, но
# он неэффективен для снижения  смещения модели, в этом случае необходимо выбирать ансамбль классификаторов с низким смещением, например,
# неподрезанные деревья решения.
