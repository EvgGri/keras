
# Ядерный метод главных компонент, для нелинейных отображений

# В реальных задачах часто случается так, что не существует линейной разделимости входных данных.
# Ядерный метод PCA позволит преобразовывать линейно неразделимые данные на новое подпространство более низкой размерности, которое подходит для
# линейных классификаторов.

# Загрузка данных - полумесяцы / линейно не разделимы
# Данные должны быть обязательно стандартизированными, с нулевым мат. ожиданием и единичной дисперсией
from sklearn.datasets import make_moons
X,y = make_moons(n_samples=100, random_state=123)

import matplotlib.pyplot as plt
plt.scatter(X[y==0,0], X[y==0,1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1,0], X[y==1,1], color='blue', marker='o', alpha=0.5)
plt.show()

# Ядерный метод анализа главных компонент в scikit-learn
from sklearn.decomposition import KernelPCA
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma = 15)
X_skernpca = scikit_kpca.fit_transform(X)

plt.scatter(X_skernpca[y==0,0], X_skernpca[y==0,1], color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y==1,0], X_skernpca[y==1,1], color='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.xlabel('PC2')
plt.show()

# =-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Пример. Разделение концентрических кругов

# Загрузка данных
# Данные должны быть обязательно стандартизированными, с нулевым мат. ожиданием и единичной дисперсией
from sklearn.datasets import make_circles
X,y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

import matplotlib.pyplot as plt
plt.scatter(X[y==0,0], X[y==0,1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1,0], X[y==1,1], color='blue', marker='o', alpha=0.5)
plt.show()
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma = 15)
X_skernpca = scikit_kpca.fit_transform(X)

plt.scatter(X_skernpca[y==0,0], X_skernpca[y==0,1], color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y==1,0], X_skernpca[y==1,1], color='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.xlabel('PC2')
plt.show()
