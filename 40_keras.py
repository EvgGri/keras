# -=-=-=-=-=-=-=-=- Загрузка данных MNIST

import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
    print('Загрузить данные Mnist из пути Path\n')
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype = np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype = np.uint8).reshape(len(labels), 784)

    return images, labels

X_train , y_train = load_mnist('/Users/grigorev-ee/Work/AnacondaProjects/My_projects/ml_w_python/data/mnist/', kind = 'train')

print('Тренировка - строки: %d, столбцы: %d' % (X_train.shape[0], X_train.shape[1]))

X_test , y_test = load_mnist('/Users/grigorev-ee/Work/AnacondaProjects/My_projects/ml_w_python/data/mnist/', kind = 't10k')
print('Тестирование - строки: %d, столбцы: %d' % (X_test.shape[0], X_test.shape[1]))

# -=-=-=-=-=-=- Реализация многослойного персептрона на Keras

# Приведем массив изображений MNIST в 32-разрядный формат:
import theano
theano.config.floatX='float32'

X_train=X_train.astype(theano.config.floatX)
X_test=X_test.astype(theano.config.floatX)

# Далее необходимо преобразовать метки классов (целые числа от 0 до 9) в формат прямого кода
import tensorflow as tf
import tensorflow

from tensorflow import keras
from keras.utils import np_utils
print('Первые 3 метки:', y_train[:3])

y_train_ohe=np_utils.to_categorical(y_train)
print('\nПервые 3 метки (прямой код):\n', y_train_ohe[:3])
 # Будем использовать туже архитектуру нейронных сетей, как для задачи распознавания изображений.
 # Заменим логистические узлы в скрытом слое на функции активации в виде гиперболического тангенса, заменим логистическую функцию в
 # выходном слое на функцию softmax и добавим дополнительный скрытый слой.

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

np.random.seed(1)

# Инициализируем новую модель для реализации нейронной сети с прямым распространением сигналов.
model=Sequential()

# Послен этого мы можем добавить в модель столько слоев, сколько нам необходимо. Однако, учитывая, что первый добавленный нами слой
# является входным, мы должны удостовериться, что атрибут input_dim соотвествует числу признаков (столбцов) в тренировочном наборе
# (в данном случае 768).
# Кроме того, мы должны удостовериться, что число выходных узлов (output_dim) и число входных узлов (input_dim) двух последовательных
# слоев соответствуют друг другу.
model.add(Dense(input_dim=X_train.shape[1], output_dim=50, init='uniform', activation='tanh'))
# В предыдущем примере мы мы добавили два скрытых слоя с 50 узлами плюс 1 узел смещения каждый. Отметим, что в библиотеке Keras узел
# смещения в полносвязных сетях инициализируется нулем.
model.add(Dense(input_dim=50, output_dim=50, init='uniform', activation='tanh'))
# Наконец, число узлов в выходном слое должно быть равно числу уникальных меток классов - числу столбцов в массиве меток классов в прямой
# кодировке.
model.add(Dense(input_dim=50, output_dim=y_train_ohe.shape[1], init='uniform', activation='softmax'))
# Прежде, чем скомпилировать модель, мы также должны задать оптимизатор. Мы выбрали оптимизацию на основе стохастического градиентного спуска,
# с которой мы знакомы из предыдущих глав. Кроме того, мы можем задать значения для константы снижения весов и импульса обучения, чтобы
# скорректировать темп обучения в каждой эпохе.
sgd=SGD(lr=0.001, decay=1e-7, momentum=.9)

# После этого мы задаем функцию стоимости categorical_crossentropy. Бинарная перекрестная энтропия - это просто технический термин для
# функции стоимости в логистической регрессии, и категориальная перекрестная энтропия - это ее обобщение для многоклассовых прогнозов
# с использованием функции softmax
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# Just disables the warning, doesn't enable AVX/FMA // игнорирование ошибки
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# После компиляции модели можно натренировать ее путем вызова метода fit
# Здесь мы используем мини-пакетный стохастический градиент с размером пакета 300 тренировочных образцов из расчета на пакет.
# Мы тренируем MLP на 50 эпох и можем отслеживать оптимизацию функции стоимости, установив параметр verbose=1.
# Параметр validation_split используется для резервирования 10% данных для проверки после каждой эпохи, с тем чтобы мы могли удостовериться, что модель
# не находится в переподогнанном состоянии во время тренировки.
model.fit(X_train, y_train_ohe,  epochs=50, batch_size=300, verbose=1, validation_split=0.1)
# Предскажим метки классов на тренировочном наборе
y_train_pred=model.predict_classes(X_train, verbose=0)
print('Первые метки классов: ', y_train_pred[:10])

# Выведем аккуратность модели
train_acc=np.sum(y_train==y_train_pred, axis=None)/X_train.shape[0]
print('Верность на тренировочном наборе: %.2f%%' % (train_acc*100))

# Предскажим метки классов на тестовом наборе
y_test_pred=model.predict_classes(X_test, verbose=0)
print('Первые метки классов: ', y_test_pred[:10])

# Выведем аккуратность модели
test_acc=np.sum(y_test==y_test_pred, axis=None)/X_test.shape[0]
print('Верность на тренировочном наборе: %.2f%%' % (test_acc*100))

# Это простая нейронная сеть без оптимизации настроечных параметров
