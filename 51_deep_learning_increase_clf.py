# -=-=-=-=-=-=-=- Повышение качества распознавания набора CIFAR-10 путем углубления сети
# Один из способов повысить качество распознавания – определить более глубокую сеть с несколькими операциями свертки.
# В данном случае мы возьмем такую последовательность модулей:

# conv+conv+maxpool+dropout+conv+conv+maxpool

# И в конце – стандартная последовательность dense+dropout+dense. Функцией активации всегда будет ReLU.
# Вот как выглядит код определения новой сети:

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


# Один из способов повысить качество распознавания – определить более глубокую сеть с несколькими операциями свертки.
# В данном случае мы возьмем такую последовательность модулей:

# conv+conv+maxpool+dropout+conv+conv+maxpool

# И в конце – стандартная последовательность dense+dropout+dense. Функцией активации всегда будет ReLU.
# Вот как выглядит код определения новой сети:
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
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

# model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT, verbose=VERBOSE)
# score = model.evaluate(X_test, Y_test,batch_size=BATCH_SIZE, verbose=VERBOSE)
# print("Test score:", score[0])
# print('Test accuracy:', score[1])

# -=-=-=-=-=-=-=-=-=-=-=- История с графиками

history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# перечислить все данные в истории
print(history.history.keys())

score = model.evaluate(X_test, Y_test,batch_size=BATCH_SIZE, verbose=VERBOSE)

print("Test score:", score[0])
print('Test accuracy:', score[1])

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
