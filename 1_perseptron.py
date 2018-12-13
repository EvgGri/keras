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

# One vs. Rest - многоклассовая классификация методом один против остальных
# Метод позволяет передать в персептрон все классы классификации
from sklearn.linear_model import Perceptron
# max_iter - число эпох, eta0 - скорость обучения
ppn = Perceptron(max_iter = 40, eta0 = 0.1, random_state = 0)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Число ошибочно классифицированных образцов: %d' % (y_test != y_pred).sum())
# Другая наиболее часто используемая метрика оценки качества модели - accurasy или верность модели (1 - ошибка классификации)
# Графический анализ, такой, как кривые обучения, который позволяет обнаружить и предотвратить переобучение
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
# График областей решений для визуализации
from matplotlib.colors import ListedColormap

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# Можно ли линейно разделить по данным параметрам эти 3 класса
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
plot_decision_regions(X = X_combined_std, y = y_combined, clf = ppn, res=0.02)
plt.xlabel('длина лепестка [стандартизованная]')
plt.ylabel('ширина лепестка [стандартизованная]')
plt.legend(loc = 'upper left')
plt.show()

# Алгоритм обучения персептрона никогда не сходится на данных, которые не полностью линейно разделимы.
# Есть более мощные линейные классификаторы, которые сходятся к минимуму стоимости, даже, если данные линейно разделимы.
# Это происходит из-за того, что веса постоянно обновляются, а это происходит из-за того, что в каждой эпохе есть
# как минимум один не верно классифицированный образец.

# Для задачи линейной и бинарной классификации существует алгоритм логистической регресии.
