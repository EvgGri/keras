# Very Deep Convolutional Networks for Large­Scale Image Recognition

# В 2014 году был внесен интересный вклад в распознавание изображений (см. K. Simonyan, A. Zisserman «Very Deep Convolutional Networks for Large­Scale Image Recognition», 2014).
# В этой работе показано, что, увеличив число весовых слоев до 16–19, можно добиться значительного улучшения по сравнению с предшествующими конфигурациями.
# В одной из рассматриваемых моделей (D или VGG­16) было 16 слоев. Для обучения модели на наборе данных ImageNet ILSVRC­2012 (http://image-net.org/challenges/LSVRC/2012/)
# была написана программа на Java с использованием библиотеки Caffe (http://caffe.berkeleyvision.org/). Этот набор содержит изображения из 1000 классов, разбитые на три набора:
# обучающий (1.3 миллиона изображений), контрольный (50 000 изображений) и тестовый (100 000 изображений). Все изображения трехканальные размера 224 × 224.
# Для этой модели ошибка непопадания в первые 5 классов составила 7.5% на наборе ILSVRC­2012­val и 7.4% на наборе ILSVRC­2012­test.

# Веса, полученные в результате обучения модели, реализованной на Caffe, были преобразованы к виду, понятному Keras
# (см. https:// gist.github.com/baraldilorenzo/07d7802847aaad0a35d3), так что их можно загрузить в модель, которая ниже определена так же,
# как в оригинальной статье:

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

# define a VGG16 network

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())

    #top layer of the VGG net
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__ == "__main__":
    im = cv2.resize(cv2.imread('/Users/grigorev-ee/Work/AnacondaProjects/My_projects/ml_w_python/data/images/cat.jpg'), (224, 224)).astype(np.float32)
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    K.set_image_dim_ordering("th")

    # Test pretrained model
    model = VGG_16('/Users/grigorev-ee/Work/AnacondaProjects/My_projects/weights_of_models/vgg16_weights.h5')
    optimizer = SGD()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    out = model.predict(im)
    print(np.argmax(out))
