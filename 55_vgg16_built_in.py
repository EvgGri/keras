# -=-=-=-=-=-=-=-=- Использование встроенного в Keras модуля VGG-16

# Приложения Keras – это предварительно построенные и обученные глубокие модели.
# Веса автоматически загружаются при создании экземпляра модели и хранятся в каталоге ~/.keras/models/.
# Использовать встроенный код очень просто:

from keras.models import Model
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import numpy as np
import cv2


# готовая модель с предобученными на наборе imagenet весами
model = VGG16(weights='imagenet', include_top=True)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

# сделать размер таким же, как у изображений, на которых обучалась модель VGG16
im = cv2.resize(cv2.imread('steam-locomotive.jpg'), (224, 224))
im = np.expand_dims(im, axis=0)

# предсказание
out = model.predict(im)
plt.plot(out.ravel())

plt.show()
print np.argmax(out)
# должна быть напечатана категория 820 – паровоз

# Полный перечень предобученных моделей приведен на странице https://keras.io/applications/.
