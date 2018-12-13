# =-=-=-=-=-=-=-=-=-=-=- Некоторые полезные операции -=-=-=-=-=-=-=-=-=-=-=-=-=
# Ниже перечислены некоторые вспомогательные операции, включенные в Keras API. Их цель – упростить создание сетей, процесс обучения
# и сохранение промежуточных результатов.

# =-=-=-=-=-=-=-=-=-=-=- Сохранение и загрузка весов и архитектуры модели
# Для сохранения и загрузки архитектуры модели служат следующие функции:

# сохранить в формате JSON
json_string = model.to_json()
# сохранить в формате YAML
yaml_string = model.to_yaml()
# восстановить модель из JSON-файла from keras.models import model_from_json model = model_from_json(json_string)
# восстановить модель из YAML-файла model = model_from_yaml(yaml_string)

# Для сохранения и загрузки параметров модели служат следующие функции:
from keras.models import load_model
# создать HDF5-файл 'my_model.h5'
model.save('my_model.h5')
# удалить существующую модель
del model
# вернуть откомпилированную модель, идентичную исходной
model = load_model('my_model.h5')

# =-=-=-=-=-=-=-=-=-=-=- Обратные вызовы для управления процессом обучения
# Процесс обучения можно остановить, когда показатель качества перестает улучшаться. Для этого служит следующая функция обратного вызова:
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

# Историю потерь можно сохранить, определив такие обратные вызовы:
class LossHistory(keras.callbacks.Callback):
 def on_train_begin(self, logs={}):
   self.losses = []

def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))
    model = Sequential()
    model.add(Dense(10, input_dim=784, init='uniform')) model.add(Activation('softmax')) model.compile(loss='categorical_crossentropy', optimizer='rmsprop') history = LossHistory()
    model.t(X_train,Y_train, batch_size=128, nb_epoch=20, verbose=0, callbacks=[history])
    print history.losses


# =-=-=-=-=-=-=-=-=-=-=- Контрольные точки

# Контрольная точка – это процесс периодического сохранения мгновенного снимка состояния приложения, так чтобы приложение можно было
# перезапустить с последнего сохраненного состояния в случае отказа. Это бывает полезно при обучении глубоких моделей,
# которое часто занимает длительное время. Состоянием глубокой модели обучения в любой момент времени являются веса,
# вычисленные к этому моменту. Keras сохраняет веса в формате HDF5 (см. https://www.hdfgroup.org/) и предоставляет средства
# сохранения контрольной точки с помощью API обратных вызовов.

# Приведем несколько ситуаций, когда контрольная точка полезна:

# 1. Если требуется перезапускать программу с последней контрольной точки после того, как спотовый инстанс AWS Spot
# (см. http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ how-spot-instances-work.html) или вытесняемая виртуальная машина
# Google (см. https://cloud.google.com/compute/ docs/instances/preemptible) неожиданно остановилась.

# 2. Если требуется остановить обучение, например, для того чтобы проверить модель на тестовых данных, а затем продолжить с
# последней контрольной точки.

# 3. Если требуется сохранять бета­версию (с наилучшим показателем качества, например, потерей на контрольном наборе) модели, обучаемой
# на протяжении нескольких периодов.

# В первом и втором случае можно сохранять контрольную точку после каждого периода, для чего достаточно стандартного использования
# обратного вызова ModelCheckpoint. Приведенный ниже код показывает, как сохранить контрольную точку в процессе обучения
# глубокой модели в Keras:
from __future__ import division, print_function
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils
import numpy as np
import os

BATCH_SIZE = 128
NUM_EPOCHS = 20

MODEL_DIR = "/tmp"
(Xtrain, ytrain), (Xtest, ytest) = mnist.load_data() Xtrain = Xtrain.reshape(60000, 784).astype(" oat32") / 255 Xtest = Xtest.reshape(10000, 784).astype(" oat32") / 255 Ytrain = np_utils.to_categorical(ytrain, 10)
Ytest = np_utils.to_categorical(ytest, 10)
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

model = Sequential()
model.add(Dense(512, input_shape=(784,), activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# сохранить модель
checkpoint = ModelCheckpoint( lepath=os.path.join(MODEL_DIR, "model-{epoch:02d}.h5"))
model. t(Xtrain, Ytrain, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS, validation_split=0.1, callbacks=[checkpoint])

# В третьем случае нужно следить за показателем качества, например верностью или потерей, и сохранять контрольную точку, только если
# текущий показатель лучше, чем у предыдущей сохраненной версии. В Keras имеется дополнительный параметр объекта контрольной точки,
# save_best_only, которому следует присвоить значение true, если указанная функциональность необходима.

# Использование TensorBoard совместно с Keras
# Keras предлагает обратный вызов для сохранения показателей качества на обучающем и тестовом наборе, а также гистограмм активации для
# различных слоев модели:
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

# Сохраненные данные можно затем визуализировать с помощью программы TensorBoard, запущенной из командной строки:
tensorboard --logdir=/full_path_to_your_logs

# =-=-=-=-=-=-=-=-=-=-=- Использование Quiver совместно с Keras

# Далее мы будем обсуждать сверточные сети, специально предназначенные для обработки изображений.
# А сейчас дадим краткий обзор приложения Quiver (см. https://github.com/jakebi- an/quiver), полезного для интерактивной визуализации
# признаков сверточных сетей. После простой установки для его использования достаточно одной строки:

pip install quiver_engine
from quiver_engine import server server.launch(model)

# Эта команда запускает сервер визуализации на порту localhost:5000. Quiver позволяет визуально исследовать нейронную сеть.
