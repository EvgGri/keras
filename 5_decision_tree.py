# --Decision tree / деревья решений

# Используя алгоритм выбора решения, мы начинаем в корне дерева и расщепляем данные по признаку, который ведет к самому большому
# приросту информации (Information Gain).
# Далее процедура расщепления повторяется, пока мы не получим однородных листов. В силу этого дерево обычно обрезается путем
# установления его максимальной глубины.

# Максимизация прироста информации
# Для того, чтобы расщепить узлы в самых информативных узлах, нам необходимо выбрать целевую функцию, которую мы хотим оптимизировать.

# Функция прироста информации:
# IG(D_p,f) = I(D_p) - sum[(N_j/N_p)*I(D_j)] , здесь суммирование идёт по индексу j
# f - признак по которому выполняется расщепление
# D_p, D_j - родительский и дочерний узлы
# I - мера неоднородности
# N_p - общее число образцов в родительском узле
# N_j - общее число образцов в j-ом дочернем узле
# Таким образом, функция прироста информации - это разница между неоднородностью в родительском узле и сумме неоднородностей в дочернем узле.

# Вместе с тем, чтобы уменьшить комбинаторное пространство поиска, в Scikit-learn реализованы бинарные деревья решений
# Родительский узел расщепляется на 2 D_left & D_right:
# IG(D_p,f) = I(D_p) - (N_left/N_p)*I(D_left) - (N_right/N_p)*I(D_right)

# В силу вышесказанного в бинарных деревьях решений обычно используется 3 меры неоднородности или критерия расщепления:
# 1. мера неоднородночти Джини I_g (мера неоднородности Джини минимизирует вероятность ошибочной классификации)
# 2. энтропия I_h (энтропицный критерий пытается максимизировать взаимную информацию в дереве)
# 3. ошибка классификации I_e (минимизурует ошибку классификации)

import matplotlib.pyplot as plt
import numpy as np

def gini(p):
    return (p)*(1-(p))+(1-p)*(1-(1-p))

def entropy(p):
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

def error(p):
    return 1 - np.max([p, 1-p])

x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]

fig =plt.figure()
ax = plt.subplot(111)

for i , lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                    ['Энтропия', 'Энтропия (шкалированная)',
                    'Неоднородность Джини', 'Ошибка классификации'],
                    ['-', '-', '--', '-.'],
                    ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label = lab, linestyle = ls, lw = 2, color = c)
ax.legend(loc='center', bbox_to_anchor=(0.5, 1.15),
          ncol=3, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1 , color= 'k', linestyle= '--')
ax.axhline(y=1.0, linewidth=1 , color= 'k', linestyle= '--')
plt.ylim([0.0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Индекс неоднородности')
plt.show()


# --Дерево решений
# Глубина 3, в качестве критерия неоднородности используется Энтропия

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

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
tree.fit(X_train_std, y_train)

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
plot_decision_regions(X_combined_std, y_combined, clf = tree, res=0.02)
plt.xlabel('длина лепестка [стандартизованная]')
plt.ylabel('ширина лепестка [стандартизованная]')
plt.legend(loc = 'upper left')
plt.show()

from sklearn . tree import export_graphviz
export_graphviz(tree, out_file='/Users/grigorev-ee/Work/AnacondaProjects/My_projects/ml_w_python/trees/tree.dot',
                feature_names=[ 'petal length' , 'petal width ' ] )

import graphviz as dot
dot.render('dot', "png", "/Users/grigorev-ee/Work/AnacondaProjects/My_projects/ml_w_python/trees/tree.dot")

from IPython.display import Image
Image("/Users/grigorev-ee/Work/AnacondaProjects/My_projects/ml_w_python/trees/tree.dot.png")
