# Классификация с максимальным зазором а основе метода опорных векторов

# Подготовка данных
# Импорт основных библиотек
from sklearn import datasets
import numpy as np

# прогружаем стандартную библиотеку
iris = datasets.load_iris()

# длина и ширина лепестков цветка ириса
X = iris.data[:,[2,3]]
# метки классов, которые присутствуют
y = iris.target
# все закодировано в числовом формате для производительности
print(np.unique(y))

# оценка модели на ранее не встречавшихся данных
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# стандартизация признаков из модуля preprocessing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# вычисление параметров распределения для train данных (стандартное отклонение и мат. ожидание)
# для каждого набора признаков. После вызова trasform мы стандартизируем тестовые и тренинговые данные.
# Для стандартизации тестового набора мы используем теже самые параметры, вычисленные для train набора.
# Поэтому значения в тренировочном и тестовом наборе сопоставимы.
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# -- Непосредственно анализ данных

# Support Vector Machine, можно рассматривать, как расширение персептрона
# Используя алгоритм персептрона мы минимизировали ошибки классификации, в SVM задача оптимизации - максимизировать зазор.
# Зазор - это расстояние меджу разделяющей гиперплоскостью (границей решения) и самыми близкими к этой гиперплоскости тренировочными образцами,
# так называемыми опорными векторами.

# Описание наличия границ решения с большими зазорами, является то, что такие модели имеют более низкую ошибку обобщения.

# -- Обработка нелинейно разделимого случая при помощи ослабленных переменных.

# Вводится ослабленная (slack) переменная, которая приводит к классификации с мягким зазором.
# Мотивация введения данной переменой состоит в том, что линейные ограничения должны быть
# ослаблены для нелинейно разделимых данных, с тем чтобы зрешить сходимость оптимизации с частием случаев
# ошибочной классификации  условиях надлежащего штрафования стоимости.

# Ослабленная переменная  положительными значениями просто добавляется к линейным ограничениям.
# Параметр С используется для настройки компромисса между смещением и дисперсией, это используется для регуляризации.

# Уменьшение С увеличивает смещение модели и понижает её дисперсию.

from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0,random_state=0)
svm.fit(X_train_std, y_train)
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
plot_decision_regions(X_combined_std, y_combined, clf =svm, res=0.02)
plt.xlabel('длина лепестка [стандартизованная]')
plt.ylabel('ширина лепестка [стандартизованная]')
plt.legend(loc = 'upper left')
plt.show()

# -- Замечания
# В практических задачах классификации линейная регрессия и метод опорных векторов часто приводят к очень похожим результатам.
# Логистическая регрессия пытаются максимизировать условные вероятности тренировочных данных, что делает ее более подверженной выбросам,
# чем методы опорных векторов (SVM). Методы опорных векторов главным образом сосредоточены на точках, ближайших к границе решения.