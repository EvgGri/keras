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

# -=-=-= Отбор содержательных признаков
# Если модель работает намного лучше на тренировочном наборе данных, чем на тестовом, это говорит о сильном переобучении.
# Т.е. модель слишком близко аппроксимирует параметры слишком близко к отдельно взятым наблюдениям в тренировочном наборе данных,
# но не обобщает хорошо на реальные данные, мы говорим, что модель имеет большую дисперсию. Причина в том, что наша модель
# слишком сложна для нашего набора тренировочных данных.

# -=-=-=-= Общее решение для снижения ошибки обобщения
# 1. собрать больше тренировочных данных
# 2. ввести штраф за сложность на основе регляризации
# 3. выбрать более проустую модель с меньшим числом параметров
# 4. снизить размерность данных

# L2-регуляризация это один из подходов для снижения сложности модели, путем наложения штрафа на большие индивидуальные веса,
# где норма L2 нашего весового вектора w определяется следующим образом:
# L2: ||w||_2^2=sum_j(w_j^2) [стр. 120 с подробными комментариями]
# L1: ||w||_1=sum_j(|w_j|)
# Здесь заменяется квадрат весовых коэффициентов на сумму их абсолютных значений.

# В отличие от L2-регуляризации, L1-регуляризация создает разреженные векторы признаков, большинство весов признаков будет равно 0.
# Разреженность может быть полезна на практике, когда есть высокоразмерный набор данных с большим числом нерелевантных (лишних) признаков.
# В этом смысле L1-регуляризация может пониматься как метод отбора признаков.

# Наша задача найти комбинацию весовых коэффициентов, которые минимизируют функцию стоимости для тренировочных данных.
# Теперь регуляризация может быть представлена как добавление к функции стоимости штрафного члена, который поощряет веса с меньшими
# значениями. Таким образом, увеличивая силу регуляризации параметром регуляризации lambda, мы стягиваем веса к нулю и уменьшаем зависимость
# модели от тренировочных данных. (страница 120)
# L2-регуляризация - это круг на странице, L1-ромб, т.к. там идёт просто сумма.

# Подробно расписывается отличие L1 от L2 регуляризации в книге "Элементы статического обучения" Хасти, Тибширани

# Чтобы для упорядоченных моделей, которые поддерживают L1-регуляризацию, получить разреженное решение, можно установить
# (в библиотеке Scikit-Learn) параметр penalty равным l1:
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)
print('Верность на тренировочном наборе:', lr.score(X_train_std, y_train))
print('Верность на тестовом наборе:', lr.score(X_test_std, y_test))

# Мы выполняем подгонку объекта LogisticRegression на многоклассовом наборе данных, подход OVR - один против всех,
# используется по умолчанию, при этом первое значение точки пересечения принадлежит модели, которая оптимизирована под первый класс, и т.д.
lr.intercept_
# Массив весов, по одному весовому вектору для каждого класса. Каждая строка состоит из 13 весов, где каждый вес помножен на
# соответствующий признак в 13-мерном наборе данных сортов вин для вычисления чистого входа
lr.coef_
# Отмечаем для себя, что весовые векторы разрежены, т.е. имеют всего несколько ненулевых записей.

# Построим график траекторий регуляризации, т.е.  весовых коэффициентов, разных признаков для разных сил регуляризации.
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4, 6, dtype=float):
    lr = LogisticRegression(penalty='l1',
                            C = 10 ** c,
                            random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column],
             color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5),10**5])
plt.ylabel('весовой коэффициент')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',
          bbox_to_anchor=(1.38,1.03),
          ncol=1, fancybox=True)
plt.show()
# Итоговый график говорит о том, что все веса признаков будут обнулены, если штрафовать модель сильным параметром регуляризации
# C < 0.1 , где C - инверсия параметра регуляризации.