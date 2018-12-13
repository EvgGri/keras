# -=-=-=-=-=-=-= Применение Keras adversarial для создания ПСС, подделывающей MNIST

# Keras adversarial (https://github.com/bstriner/keras-adversarial) – написанный на Python пакет с открытым исходным кодом, предназначенный для построения ПСС.
# Его автор – Бен Страйнер (Ben Striner) (https://github.com/bstriner и https://github.com/bstriner/ keras-adversarial/blob/master/LICENSE.txt).
# Поскольку версия Keras 2.0 появилась совсем недавно, я рекомендую скачать самую последнюю версию этого пакета:

# git clone --depth=50 --branch=master https://github.com/bstriner/keras-adversarial.git

# И установить его:
# python setup.py install

# from keras.applications.inception_v3 import InceptionV3
# from keras.preprocessing import image
# from keras.models import Model
# from keras.layers import Dense, GlobalAveragePooling2D
# from keras import backend as K

# Если генератор G и дискриминатор D основаны на одной и той же модели M, то их можно объединить в состязательную модель; она получает тот же вход, что и M,
# но цели и показатели качества различны для G и D. В библиотеке определена следующая функция создания модели:
# adversarial_model = AdversarialModel(base_model=M, player_params=[generator.trainable_weights, discriminator.trainable_weights],
#                                     player_names=["generator", "discriminator"])

# Если генератор G и дискриминатор D основаны на разных моделях, то можно воспользоваться такой функцией:
# adversarial_model = AdversarialModel(player_models=[gan_g, gan_d], player_params=[generator.trainable_weights, discriminator.trainable_weights],
#                                      player_names=["generator", "discriminator"])

# -=-=-=-=-=-=-=-=-= Рассмотрим пример вычислений для MNIST:

import matplotlib as mpl

# Эта строка позволяет использовать mpl без определения DISPLAY
mpl.use('Agg')

# Ниже рассматривается открытый исходный код
# (https://github. com/bstriner/keras-adversarial/blob/master/examples/example_gan_convolutional.py).
# В нем используется синтаксис Keras 1.x, но код работаети с Keras 2.x благодаря набору вспомогательных функций в файле legacy.py.
# Содержимое файла legacy.py приведено в приложении, а также по адресу
# https://github.com/bstriner/keras-adversarial/blob/ master/keras_adversarial/legacy.py

# Сначала импортируется ряд модулей. Мы уже встречались со всеми, кроме LeakyReLU, специальной версии ReLU, которая допускает малый
# градиент, когда нейрон не активен. Экспериментально показано, что в ряде случаев функция LeakyReLU может улучшить качество ПСС
# (см. B. Xu, N. Wang, T. Chen, M. Li «Empirical Evaluation of Recti ed Activations in Convolutional Network», arXiv:1505.00853, 2014).
from keras.layers import Dense, Reshape, Flatten, Dropout, LeakyReLU, Input, Activation, BatchNormalization
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras.datasets import mnist

import pandas as pd
import numpy as np

# Затем импортируются специальные модули для ПСС
import keras.backend as K
from keras_adversarial.legacy import Dense, BatchNormalization, Convolution2D
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
# Не заработало, пробуем ввести функции явно
# from image_utils import dim_ordering_fix, dim_ordering_input, dim_ordering_reshape, dim_ordering_unfix

import keras.backend as K
import numpy as np
from keras.layers import Input, Reshape


def dim_ordering_fix(x):
    if K.image_dim_ordering() == 'th':
        return x
    else:
        return np.transpose(x, (0, 2, 3, 1))


def dim_ordering_unfix(x):
    if K.image_dim_ordering() == 'th':
        return x
    else:
        return np.transpose(x, (0, 3, 1, 2))


def dim_ordering_shape(input_shape):
    if K.image_dim_ordering() == 'th':
        return input_shape
    else:
        return (input_shape[1], input_shape[2], input_shape[0])


def dim_ordering_input(input_shape, name):
    if K.image_dim_ordering() == 'th':
        return Input(input_shape, name=name)
    else:
        return Input((input_shape[1], input_shape[2], input_shape[0]), name=name)


def dim_ordering_reshape(k, w, **kwargs):
    if K.image_dim_ordering() == 'th':
        return Reshape((k, w, w), **kwargs)
    else:
        return Reshape((w, w, k), **kwargs)


def channel_axis():
    if K.image_dim_ordering() == 'th':
        return 1
    else:
        return 3

# Состязательные модели обучаются в ходе игры с несколькими игроками.
# Если дана базовая модель с n целями и k игроками, то создается модель с n*k целями, в которой каждый игрок оптимизирует потерю
# на своих целях. Кроме того, функция simple_gan порождает ПСС с заданными целями gan_targets.
# Отметим, что в библиотеке цели для генератора и дискриминатора противоположны, это стандартная практика для ПСС:
def gan_targets(n):
    """
    Стандартные цели обучения
    [generator_fake, generator_real, discriminator_fake, discriminator_real] = [1, 0, 0, 1]
    :param n: число примеров :return: массив целей
    """
    generator_fake = np.ones((n, 1))
    generator_real = np.zeros((n, 1))
    discriminator_fake = np.zeros((n, 1))
    discriminator_real = np.ones((n, 1))
    return [generator_fake, generator_real, discriminator_fake, discriminator_real]

# Генератор в этом примере определяется так же, как мы видели раньше. Но теперь мы используем функциональный синтаксис –
# каждый модуль в конвейере просто передается в качестве параметра следующему модулю.
# Первый слой сети плотный, инициализирован в режиме glorot_normal. В этом режиме используется гауссов шум, масштабированный
# на сумму входов и выходов из узла. Аналогично инициализированы все остальные модули. Параметр mode=2 функции BatchNormalization
# определяет попризнаковую нормировку на основе статистики каждого пакета. Экспериментально показано, что так получаются более
# качественные результаты:
def model_generator():
    nch = 256
    g_input = Input(shape=[100])
    H = Dense(nch * 14 * 14, init='glorot_normal')(g_input)
    # H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)
    H = dim_ordering_reshape(nch, 14)(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Convolution2D(int(nch / 2), 3, 3, border_mode='same',
    init='glorot_uniform')(H)
    # H = BatchNormalization(mode=2, axis=1)(H)
    H = Activation('relu')(H)
    H = Convolution2D(int(nch / 4), 3, 3, border_mode='same',
    init='glorot_uniform')(H)
    # H = BatchNormalization(mode=2, axis=1)(H)
    H = Activation('relu')(H)
    H = Convolution2D(1, 1, 1, border_mode='same',
    init='glorot_uniform')(H)
    g_V = Activation('sigmoid')(H)
    return Model(g_input, g_V)

# Дискриминатор очень похож на определенный нами выше. Единственное различие – модуль LeakyReLU:
def model_discriminator(input_shape=(1, 28, 28), dropout_rate=0.5):
    d_input = dim_ordering_input(input_shape, name="input_x")
    nch = 512
    H = Convolution2D(int(nch / 2), 5, 5, subsample=(2, 2),
    border_mode='same', activation='relu')(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Convolution2D(nch, 5, 5, subsample=(2, 2),
    border_mode='same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Flatten()(H)
    H = Dense(int(nch / 2))(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    d_V = Dense(1, activation='sigmoid')(H)
    return Model(d_input, d_V)

# Далее следуют две простые функции для загрузки и нормировки набора данных MNIST:
def mnist_process(x):
    x = x.astype(np. oat32) / 255.0
    return x

def mnist_data():
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    return mnist_process(xtrain), mnist_process(xtest)

# На следующем шаге определяется совместная модель ПСС в виде комбинации генератора и дискриминатора.
# Заметим, что веса инициализируются функцией normal_latent_sampling, которая производит выборку из нормального распределения:
if __name__ == "__main__":
    # z принадлежит R^100
    latent_dim = 100
    # x принадлежит R^{28x28}
    input_shape = (1, 28, 28)
    # генератор (z -> x)
    generator = model_generator()
    # дискриминатор (x -> y)
    discriminator = model_discriminator(input_shape=input_shape)
    # ПСС (x - > yfake, yreal), z генерируется на GPU
    gan = simple_gan(generator, discriminator, normal_latent_sampling((latent_dim,)))
    # печатать общие сведения о моделях
    generator.summary()
    discriminator.summary()
    gan.summary()

# Затем в примере создается ПСС и компилируется модель, обученная с использованием оптимизатора Adam и функции потерь binary_crossentropy:

# построить состязательную модель
model = AdversarialModel(base_model=gan,player_params=[generator.trainable_weights, discriminator.trainable_weights],player_names=["generator", "discriminator"])
model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),player_optimizers=[Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4)], loss='binary_crossentropy')
