# Алгоритмы последовательного отбора признаков

# Эти алгоритмы принадлежат к семейству жадных алгоритмов поиска, которые используя информацию из d-мерного пространства признаков,
# переходят к k-мерному пространству признаков, где d<k.
# Основная мотивация алгоритмов - отобрать наиболее релевантные признаки для решения задачи, чтобы увеличить вычислительную эффективность
# и уменьшить ошибку обобщения модели.

# Используя случайный лес, можно измерить важность признака как усредненное уменьшение неоднородности, вычисленное из всех деревьев в лесе,
# не делая никаких допущений по поводу наличия или отуствия линейной разделимости данных.

# Удобно то, что в библиотеке Scikit-Learn для леса решений уже реализовано аккумулирование важности признаков, и к ним можно обращаться
# через атрибут feature_importances_ после подгонки классификатора RandomForestClassifier


# =-=-=-=-=-=-=-=-=- Загрузка данных -=-=-=-=-=-=-=-=-

# Разбивка на тренировочное и тестовое подмножество

import pandas as pd
import numpy as np

from sklearn.datasets import load_wine
data = load_wine()
data.target[[10, 80, 140]]
list(data.target_names)


df_wine = pd.DataFrame(data.data)

df_wine.columns=['Алкоголь','Яблочная кислота','Зола',
                 'Щелочность золы','Магний','Всего фенола','Флаваноиды',
                 'Фенолы нефлаваноидные','Проантоцианины','Интенсивность цвета',
                 'Оттенок','OD280 / OD315 разбавленных вин','Пролин']
df_wine.head()
df_wine['Метка класса'] = pd.DataFrame(data.target)
df_wine.describe()
# Размер data frame'a
df_wine.shape

# Разбиение на train и test выборки
X, y = df_wine.iloc[:,0:13].values, df_wine['Метка класса'].values

# Альтернативная задача массива данных
# X = df_wine.ix[:,['Алкоголь','Яблочная кислота','Зола',
#                  'Щелочность золы','Магний','Всего фенола','Флаваноиды',
#                  'Фенолы нефлаваноидные','Проантоцианины','Интенсивность цвета',
#                  'Оттенок','OD280 / OD315 разбавленных вин','Пролин']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Масштабирование признаков(нормирование, приведение к диапазону 0-1)
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# С практической точки зрения лучше использовать стандартизацию признаков (приведеение к нормальному распределению с единичной дисперсией).
# Причина в том, что многие линейные модели, такие, как логистическая регрессия и метод опорных векторов инициализируют веса нулями или
# близкими к 0 значениями, но вид нормального распределения упрощает извлечение весов.

# Кроме того, стандартизация содержит полезную информацию о выбросах и делает алгоритм менее чувствительным к выбросам, в отличие от
# минимаксного масштабирования, которое шкалирует данные в ограниченном диапазоне значений.

# стандартизация признаков из модуля preprocessing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# вычисление параметров распределения для train данных (стандартное отклонение и мат. ожидание)
# для каждого набора признаков. После вызова trasform мы стандартизируем тестовые и тренинговые данные.
# Для стандартизации тестового набора мы используем теже самые параметры, вычисленные для train набора.
# Поэтому значения в тренировочном и тестовом наборе сопоставимы.
sc.fit(X_train)
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# =-=-=-=-=-=-=-=-=- Непосредственная реализация отбора признаков через RandomForestClassifier -=-=-=-=-=-=-=-=-

# Натренируем лес из 10 000 деревьев на нашем наборе данных и упорядочим 13 признаков по их соответствующим мерам важности.
# Отметим, что модели на основе деревьев решений в стандартизации и нормализации не нуждаются.
# Но мы проверим, изменится ли важность признаков для стандартизованного набора данных или нет.
from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns

forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("{:3d})".format(f), "{:25.25}".format(feat_labels[indices[f]]), "{:10.6f}".format(importances[indices[f]]))

# Выведем график, в котором разные признаки из набора данных упорядочены по их относительной важности,
# отметим то, что важности признаков нормализованы, т.е. в сумме они дают единицу.
import matplotlib.pyplot as plt
plt.title('Важности признаков')
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align = 'center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation = 90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

# =-=-=-=-=-=-=-=-=- Непосредственная реализация отбора признаков через RandomForestClassifier на стандартизованных данных -=-=-=-=-=-=-=-=-

# Натренируем лес из 10 000 деревьев на нашем наборе данных и упорядочим 13 признаков по их соответствующим мерам важности.
# Отметим, что модели на основе деревьев решений в стандартизации и нормализации не нуждаются.
# Но мы проверим, изменится ли важность признаков для стандартизованного набора данных или нет.
from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns

forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train_std, y_train)

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X_train_std.shape[1]):
    print("{:3d})".format(f), "{:30.30}".format(feat_labels[indices[f]]), "{:10.6f}".format(importances[indices[f]]))

# Выведем график, в котором разные признаки из набора данных упорядочены по их относительной важности,
# отметим то, что важности признаков нормализованы, т.е. в сумме они дают единицу.
import matplotlib.pyplot as plt
plt.title('Важности признаков')
plt.bar(range(X_train_std.shape[1]), importances[indices], color='lightblue', align = 'center')
plt.xticks(range(X_train_std.shape[1]), feat_labels[indices], rotation = 90)
plt.xlim([-1, X_train_std.shape[1]])
plt.tight_layout()
plt.show()
# Видим, что важность признаков стандартизованном наборе данных не изменилась.

# Интересно то, что три находящихся на вершине рейтинга признака в приведенном выше графике также находятся среди пяти лучших признаков,
# отобранных в результате выполнения алгоритма SBS.
#
# Однако, что касается интерпритируемости, метод случайных лесов несет в себе один важный изъян.
# Например, если два или более признаков высоко коррелируют, то один из признаков может получить очень высокую степень важности, тогда как
# информация о другом признаке может оказаться не захваченной полностью.

# С другой стороны, нам не нужно беспокоиться по этому поводу, если нас интересует только предсказательная способность модели, а не интерпретация
# важности признаков.

# В Scikit-Learn также реализован метод transform, который отбирает признаки, основываясь на определенном пользователем пороге после подгонки
# модели. Он часто применяется при использовании RandomForestClassifier в качестве селектора признаков.

# К примеру, можно установить порог в 0.15 для сведения набора данных к 3 наиболее важных признаков.

# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.15
from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(forest, threshold=0.15)

# Train the selector
sfm.fit(X_train_std, y_train)

# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(feat_labels[feature_list_index])

#Create A Data Subset With Only The Most Important Features

# There are indeed several ways to get feature "importances". As often, there is no strict consensus about what this word means.
# In scikit-learn, we implement the importance as described in [1] (often cited, but unfortunately rarely read...). It is sometimes called "gini importance" or "mean decrease impurity" and is defined as the total decrease in node impurity (weighted by the probability of reaching that node (which is approximated by the proportion of samples reaching that node)) averaged over all trees of the ensemble.
# In the literature or in some other packages, you can also find feature importances implemented as the "mean decrease accuracy". Basically, the idea is to measure the decrease in accuracy on OOB data when you randomly permute the values for that feature. If the decrease is low, then the feature is not important, and vice-versa.
# [1]: Breiman, Friedman, "Classification and regression trees", 1984.

# Transform the data to create a new dataset containing only the most important features
# Note: We have to apply the transform to both the training X and test X data.
X_important_train = sfm.transform(X_train_std)
X_important_test = sfm.transform(X_test_std)


# Train A New Random Forest Classifier Using Only Most Important Features
# Create a new random forest classifier for the most important features
forest_important = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the new classifier on the new dataset containing the most important features
forest_important.fit(X_important_train, y_train)

# Compare The Accuracy Of Our Full Feature Classifier To Our Limited Feature Classifier

# Apply The Full Featured Classifier To The Test Data
y_pred = forest.predict(X_test_std)

from sklearn.metrics import accuracy_score
# View The Accuracy Of Our Full Feature (4 Features) Model
accuracy_score(y_test, y_pred)


# Apply The Full Featured Classifier To The Test Data
y_important_pred = forest_important.predict(X_important_test)

# View The Accuracy Of Our Limited Feature (2 Features) Model
accuracy_score(y_test, y_important_pred)
