# =-=-=-=-=-=-=-=-=-=-=- Пример ГСНС – LeNet

# Ян Лекун (Yann le Cun) предложил (см. статью Y. LeCun, Y. Bengio «Convolutional Networks for Images, Speech, and Time­Series», Brain
# Theory Neural Networks, vol. 3361, 1995) семейство сверточных сетей, получившее название LeNet, обученных распознаванию рукописных
# цифр из набора MNIST и устойчивых к простым геометрическим преобразованиям и искажению.
# Основная идея состоит в наличии чередующихся слоев, реализующих операции свертки и max­ пулинга. Операции свертки основаны на тщательно
# подобранных локальных рецептивных полях с весами, разделяемыми между несколькими картами признаков.
# Последние слои полносвязные – как в традиционном МСП со скрытыми слоями и функцией активации softmax в выходном слое.

# Для определения сети LeNet в Keras используется модуль двумерной сверточной сети:
from tensorflow import keras

from keras.layers.convolutional import Conv2D
filters=2
kernel_size=3
# keras.layers.convolutional.Conv2D( lters, kernel_size, padding='valid')
keras.layers.Conv2D(filters, kernel_size, padding='valid')

# Здесь  filters – число сверточных ядер (например, размерность выхода), kernel_size – одно целое число или кортеж (либо список)
# из двух целых чисел, задающих ширину и высоту двумерного окна свертки (если указано одно число, то ширина и высота одинаковы),
# а padding='same' означает, что используется дополнение. Существует два режима: padding='valid' означает, что свертка вычисляется
# только там, где фильтр целиком помещается в области входа, поэтому выход оказывается меньше входа, а padding='same' – что размер
# выхода такой же (same), как размер входа, для чего входная область дополняется нулями по краям.

from keras.layers import Convolution2D as Conv2D
from keras.layers.convolutional import Deconv2D as Conv2DTranspose

# keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
# Здесь pool_size=(2, 2) – кортеж из двух целых чисел, определяющих коэффициенты уменьшения изображения по вертикали и по горизонтали.
# Таким образом, (2, 2) означает, что изображение уменьшается вдвое в обоих направлениях.
# Наконец, параметр strides=(2, 2) определяет шаг обработки.
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Непосредственно реализация

from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import matplotlib.pyplot as plt

# Затем определяется сеть LeNet
np.random.seed(1671)  # for reproducibility

#define the convnet
class LeNet:
	@staticmethod
	def build(input_shape, classes):

        # Первый слой – сверточный с функцией активации ReLU, за ним следует слой max­ пулинга. В нашей сети будет 20 сверточных фильтров размера 5 × 5.
        # Размер выхода такой же, как размер входа – 28 × 28. Поскольку первым элементом конвейера является модуль Convolution2D, необходимо определить его форму, input_shape.
        # Операция max ­пулинга реализует скользящее окно, перемещающееся по слою, и вычисляет максимальное значение в области. Шаг перемещения по горизонтали и по вертикали равен 2.
		model = Sequential()
		# CONV => RELU => POOL
		model.add(Conv2D(20, kernel_size=5, padding="same",
			input_shape=input_shape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Затем добавляется второй сверточный слой с функцией активации ReLU, а за ним еще один слой max­пулинга.
        # Но теперь мы увеличиваем число сверточных фильтров с 20 до 50.
        # Увеличение числа фильтров в более глубоких слоях – стандартный прием глубокого обучения.
		# CONV => RELU => POOL
		model.add(Conv2D(50, kernel_size=5, padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Затем идет довольно стандартный слой уплощения, плотный слой с 500 нейронами и softmax­классификатор с 10 классами
		# Flatten => RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		# a softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model

# сеть и ее обучение
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT=0.2

IMG_ROWS, IMG_COLS = 28, 28 # input image dimensions
NB_CLASSES = 10  # number of outputs = number of digits
INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)

# данные: перетасованы и разбиты на обучающий и тестовый набор
(X_train, y_train), (X_test, y_test) = mnist.load_data()
K.set_image_dim_ordering("th")

# рассматриваем как числа с плавающей точкой и нормируем
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# нам нужна форма 60K x [1 x 28 x 28], подаваемая на вход сверточной сети
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# преобразуем векторы классов в бинарные матрицы классов
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# инициализировать оптимизатор и модель
model = LeNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
	metrics=["accuracy"])

history = model.fit(X_train, y_train,
		batch_size=BATCH_SIZE, epochs=NB_EPOCH,
		verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])

# перечислить все данные в истории
print(history.history.keys())

# построить график изменения верности
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# построить график изменения потери
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Как видим, время заметно выросло, теперь на каждую итерацию обучения сети уходит ~134 секунды вместо ~1–2 секунд для сети из главы 1.
# Однако и максимальная верность теперь равна 99.96%.

# Из графиков верности и потери видно, что для достижения верности 99.2% было бы достаточно всего 4–5 итераций.
