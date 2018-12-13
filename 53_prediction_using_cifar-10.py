# -=-=-=-=-=-=-=-=-=- Предсказание на основе результатов обучения на наборе CIFAR-10

# Пусть теперь мы хотим использовать обученную на наборе CIFAR­10 модель для массовой обработки изображений.
# Поскольку мы сохранили модель вместе с весами, то обучать ее каждый раз не нужно.

import numpy as np
import scipy.misc
from keras.models import model_from_json
from keras.optimizers import SGD

# загрузить модель
model_architecture = 'cifar10_architecture.json'
model_weights = 'cifar10_weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

# загрузить изображения
img_names = ['/Users/grigorev-ee/Work/AnacondaProjects/My_projects/ml_w_python/data/images/cat-standing.jpg', '/Users/grigorev-ee/Work/AnacondaProjects/My_projects/ml_w_python/data/images/dog.jpg']
imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (32,32)),(1, 0, 2)).astype('float32') for img_name in img_names]
imgs = np.array(imgs) / 255

# обучить
optim = SGD()
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

# предсказать
predictions = model.predict_classes(imgs)
print(predictions)
