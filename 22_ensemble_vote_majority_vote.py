from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

iris = datasets.load_iris()
X, y = iris.data[50:,[1,2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# Натренируем три разных классификатора - классификатор на основе логистической регрессии, дерева решений и к-ближайших соседей и посмотрим
# на их индивидуальные качественные характеристики путем 10-блочной перекрестной проверки на тренировочном наборе данных, перед тем, как
# объединить их в ансамблевый классификатор.
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np

clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=0)
clf2 = DecisionTreeClassifier(max_depth=1, criterion="entropy", random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])

clf_labels = ['Логистическая регрессия', 'Дерево решений', 'К ближайших соседей']
print('10-блочная перекрестная проверка:\n')

for clf, label in zip([pipe1,clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf,
                                X=X_train,
                                y=y_train,
                                cv=10,
                                scoring='roc_auc')
    print("ROC/AUC: {:.2f} (+/- {:.2f}) {:5}".format(scores.mean(), scores.std(), label))

print('* ROC AUC = площадь под ROC-кривой')

# -=-=-=-=-=-=-=-=-=-=-=-=- Реализация класса -=-=-=-=-=-=-=-=-
# Реализация классификатора на основе мажоритарного голосования
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """Ансамблевый классификатор на основе мажоритарного голосования
    Параметры:
    classifiers - классификаторы ансамбля, форма = [n_classifiers]
    vote - str, ['classlabel', 'classprobability'], если метка classlabel, то берется argmax меток классов, если classprobability, то
    для прогнозирования метки используется argmax сумму вероятностей
    weighta - веса, важность классификаторов, по умолчанию None
    """

    def __init__(self, classifiers, vote='classlabel', weights = None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """Выполнить подгонку классификаторов
        Х - разреженная матрица с тренировочными образцами [n_samples, n_features]
        y - вектор целевых меток классов [n_samples]

        Возвращает объект self
        """
        # Использовать объект LabelEncoder, чтобы гарантировать, что
        # метки классов начинаются с 0, что важно для
        # вызова np.argmax в self.predict

        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)

        return self

    def predict(self, X):
        """Спрогнозировать метки классов для Х
        Х - разреженная матрица с тренировочными образцами [n_samples, n_features]

        Возвращает
        maj_vote - массив [n_samples], спрогнозированные метки классов
        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis = 1)
        else: #Голосование 'classlabel'

            #Аккумулировать результаты из вызовов clf.predict
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights = self.weights)), axis = 1, arr = predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        """Спрогнозировать вероятность класса для Х

        Параметры:
        Х - тренировочные векторы, [n_samples, n_features]

        Возвращает:
        avg_proba - взвешенная средняя вероятность для каждого класса в рассчете на образец, [n_samples, n_classes]
        """

        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis = 0, weights = self.weights)

        return avg_proba

    def get_params(self, deep = True):
        """Получить имена парметров классификатора для GridSearch"""

        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep = False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep = True)):
                    out['%s %s' % (name, key)] = value
            return out

mv_clf = MajorityVoteClassifier(classifiers=[pipe1,clf2, pipe3])
clf_labels+= ['Мажоритарное голосование']
all_clf = [pipe1, clf2, pipe3, mv_clf]

for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print("ROC_AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# Оценка и тонкая настройка ансамблевого классификатора
# Вычислим ROC-кривые на тестовом наборе данных, чтобы убедиться, что классификатор `MajorityVoteClassifier хорошо обобщается
# на ранее не встречавшиеся данные
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']

for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    # принимая, что метка положительного класса равна 1
    y_pred=clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=clr, linestyle=ls, label='%s (auc=%0.2f)' % (label, roc_auc))

plt.legend(loc='lower right')
plt.plot([0,1], [0,1], linestyle='--', color='gray', linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()
plt.xlabel('Доля ложноположительных')
plt.ylabel('Доля истинно положительных')
plt.show()

# -=-=-=-=-==-=-= Визуализация на 2-х переменных
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
from itertools import product
x_min = X_train_std[:,0].min()-1
x_max = X_train_std[:,0].max()+1
y_min = X_train_std[:,1].min()-1
y_max = X_train_std[:,1].max()+1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(7,5))

for idx, clf, tt in zip(product([0,1],[0,1]), all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0], X_train_std[y_train==0, 1], c='blue', marker='^', s=50)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==1, 0], X_train_std[y_train==1, 1], c='red', marker='o', s=50)
    axarr[idx[0], idx[1]].set_title(tt)
plt.text(-3.5, -4.5, s='Главная компонента №1', ha='center', va='center', fontsize=12)
plt.text(-10.5, 4.5, s='Главная компонента №2', ha='center', va='center', fontsize=12, rotation=90)
plt.show()
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Какие параметры использовали алгоритмы?
mv_clf.get_params()
# -=-=-=-=-=-=-=-=- Тонкая настройка гиперпараметров модели
from sklearn.model_selection import GridSearchCV
params = {'decisiontreeclassifier__max_depth': [1,2], 'pipeline-1__clf__C': [0.001, 0.1, 100.0]}
grid = GridSearchCV(estimator=mv_clf, param_grid=params, cv=10, scoring='roc_auc')
# Поиск по сетке параметров, вычисленные в результате 10-блочной перекрестной проверки
grid.fit(X_train, y_train)

for params, mean_score, scores in grid.grid_scores_:
    print("%0.3f +/- %0.2f %r" % (mean_score, scores.std() / 2, params))
print("Лучшие параметры: %s" % grid.best_params_)
print("Верность: %.2f" % grid.best_score_)
