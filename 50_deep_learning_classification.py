# =-=-=-=-=-=-=-=-=-=-=-=- Распознавание изображений из набора CIFAR-10 с помощью глубокого обучения
#
# Набор данных CIFAR­10 содержит 60 000 цветных изображений размера 32 × 32 пикселя с 3 каналами, разбитых на 10 классов.
# В обучающем наборе 50 000 изображений, в тестовом – 10 000. На следующем рисунке, взятом из репозитория CIFAR
# (https://www. cs.toronto.edu/~kriz/cifar.html), представлены случайно выбранные примеры из каждого класса.
#
# Задача состоит в том, чтобы распознать не предъявлявшиеся ра- нее изображения и отнести их к одному из 10 классов.

# Прежде всего импортируем ряд модулей, определим некоторые константы и загрузим набор данных:
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt

# набор CIFAR_10 содержит 60K изображений 32x32 с 3 каналами
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32
# константы

BATCH_SIZE = 128
NB_EPOCH = 20
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

# загрузить набор данных
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Теперь применим унитарное кодирование и нормируем изобра- жения:
# преобразовать к категориальному виду
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# преобразовать к формату с плавающей точкой и нормировать
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# В нашей сети будет 32 сверточных фильтра размера 3 × 3. Размер выхода такой же, как размер входа, т. е. 32 × 32, а в качестве функции активации
# используется ReLU, вносящая нелинейность. Далее следует операция max­пулинга с размером блока 2 × 2 и прореживание с коэффициентом 25%:

# сеть
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Следующий элемент конвейера – плотный слой с 512 нейронами и функцией активации ReLU, а за ним прореживание с коэффициентом 50% и выходной слой softmax­
# классификации с 10 классами,по одному на категорию:

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

# Определив нейросеть, мы можем перейти к обучению модели. В данном случае мы выделяем контрольный набор, помимо обучающего и тестового.
# Обучающий набор нужен для обучения модели, контрольный – для выбора наилучшего подхода к обучению, а тес­товый – для проверки обученной модели на новых данных.
# обучение
model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT, verbose=VERBOSE)
score = model.evaluate(X_test, Y_test,batch_size=BATCH_SIZE, verbose=VERBOSE)
print("Test score:", score[0])
print('Test accuracy:', score[1])

# Сохраним еще архитектуру глубокой сети:

# сохранить модель
model_json = model.to_json()
open('cifar10_architecture.json', 'w').write(model_json)
# и веса, вычисленные в результате обучения сети
model.save_weights('cifar10_weights.h5', overwrite=True)

# Выполним программу. Сеть достигает верности на тестовом наборе 66.4% при 20 итерациях.
# Мы также построили графики верности и потери и сохранили сеть методом model.summary().
