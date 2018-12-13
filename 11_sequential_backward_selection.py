# Алгоритмы последовательного отбора признаков

# Эти алгоритмы принадлежат к семейству жадных алгоритмов поиска, которые используя информацию из d-мерного пространства признаков,
# переходят к k-мерному пространству признаков, где d<k.
# Основная мотивация алгоритмов - отобрать наиболее релевантные признаки для решения задачи, чтобы увеличить вычислительную эффективность
# и уменьшить ошибку обобщения модели.

# SBS последовательно удаляет признаки из полнопризнакового пространства. Нужно определить функционал, который мы хотим минимизировать.
# Эта функция - просто разница в качестве классификатора до удаления признака и после удаления. На каждом шаге мы удаляем тот признак,
# который приводит к наименьшей потере качества классификатора.

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

# =-=-=-=-=-=-=-=-=- Непосредственная реализация SBS -=-=-=-=-=-=-=-=-

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split # cross validation
from sklearn.metrics import accuracy_score

# В раелизации ниже, k_features - требуемое число признаков, которое мы хотим получить.
# accuracy_score оценивает качество классификации модели для подмножества признаков.
# В функции itertools.combinations сужается подмножество признаков, пока подмножество не получит требуемую размерность.
# На каждой итерации, оценка качества модели accuracy_score наилучшего подмножества накапливается в списке self.scores_,
# основываясь на создаваемом внутри тестовом наборе X_test.
# Индексы столбцов окончательного подмножества признаков назначаются списку self.indicies_, который используется методом
# transform для возврата нового массива данных со столбцами отобранных признаков.
# Вместо того, чтобы вычислять критерий явным образом внутри метода fit, просто удаляется признак, который не содержится в
# подмножестве наиболее перспективных признаков.
class SBS():
    def __init__(self, estimator, k_features,
                 scoring=accuracy_score,
                 test_size=0.25, random_state=1):
                 self.scoring=scoring
                 self.estimator=clone(estimator)
                 self.k_features=k_features
                 self.test_size=test_size
                 self.random_state=random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # Стартовая размерность пространства признаков
        dim = X_train.shape[1]

        # Индексы столбцов окончательного подмножества
        self.indicies_ = tuple(range(dim))
        self.subsets_ = [self.indicies_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indicies_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indicies_, r=dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indicies_ = subsets[best]
            self.subsets_.append(self.indicies_)
            dim -= 1

            self.scores_.append(scores[best])

        self.k_scores_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indicies_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indicies):
        self.estimator.fit(X_train[:, indicies], y_train)
        y_pred = self.estimator.predict(X_test[:, indicies])
        score = self.scoring(y_test, y_pred)
        return score

# Реализация SBS с использованием классификатора KNN библиотеки Scikit-Learn
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# График верности классификации классификатора KNN
# Видно, что верность классификатора KNN улучшилась на проверочном наборе данных, когда мы уменьшили число признаков,
# это произошло благодаря тому, что мы избавились (снизилось) от проклятия размерности, кроме того видно, что классификатор
# достиг 100% точности для k={5,6,7,8,9,10,11}
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Верность')
plt.xlabel('Число признаков')
plt.grid()
plt.show()
# Проверим, что за 5-ть признаков привели к такому хорошему качеству на проверочном наборе.
# В k5 мы получили индексы столбцов 5-ти признакового пространства и вернули соответствующие имена признаков из столбцов индекса таблицы данных DataFrame
k5 = list(sbs.subsets_[8])
print(df_wine.columns[0:][k5])

# Теперь выполним оценку качества классификатора KNN на исходном тестовом наборе данных
knn.fit(X_train_std[:, k5], y_train)
print('Верность на тренировочном наборе:', knn.score(X_train_std[:, k5], y_train))
print('Верность на тестовом наборе:', knn.score(X_test_std[:, k5], y_test))

# В итоге, при меньшем, чем на половину, числе исходных признаков, мы улучшили верность прездсказания почти на 2%.
# Кроме того, уменьшилось переобучение, которое косвенно можно измерить по малому разрыву между верностью на тренировочном и тестовом наборе данных.
