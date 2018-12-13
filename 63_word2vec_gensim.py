# -=-=-=-=-=-=-=-=-=-=- Сторонние реализации word2vec

# В предыдущих разделах было подробно рассмотрено семейство моделей word2vec. Вы понимаете, как работают модели skip­грамм и CBOW и
# как самостоятельно построить их реализацию с помощью Keras. Однако существуют готовые реализации word2vec и в предположении,
# что ваша задача не слишком сложна и не сильно отличается от типичной, имеет смысл воспользоваться одной из них и не изобретать велосипед.

# Одна такая реализация word2vec – библиотека gensim. И хотя эта книга посвящена Keras, а не gensim, мы решили включить ее обсуждение,
# поскольку Keras не поддерживает word2vec, а интеграция gensim и Keras – распространенная практика.

# Установка gensim не вызывает сложностей и подробно описана на странице https://radimrehurek.com/gensim/install.html.

# Ниже показано, как построить модель word2vec с помощью gensim и обучить ее на тексте из корпуса text8, доступном по адресу
# http://mattmahoney.net/dc/text8.zip. Этот файл содержит около 17 миллионов слов, взятых из статей википедии.
# Текст был подвергнут очистке – удалению разметки, знаков препинания и символов, не принадлежащих кодировке ASCII.
# Первые 100 миллионов знаков очищенного текста и составили корпус text8. Он часто используется в качестве примера для модели word2vec,
# потому что обучение на нем происходит быстро и дает хорошие результаты.

# Сначала импортируем необходимые пакеты:
from gensim.models import KeyedVectors
import logging
import os
from gensim import models
from gensim.models import Word2Vec, KeyedVectors
# from gensim.models import word2vec
# from cltk.vector.word2vec import get_sims

# Затем читаем поток слов из корпуса text8 и разбиваем его на предложения по 50 слов в каждом. Библиотека gensim содержит встроенный
# обработчик text8, который делает нечто подобное. Поскольку мы хотим показать, как построить модель для любого (предпочтительно большого)
# корпуса, который может и не помещаться целиком в память, то продемонстрируем порождение этих предложений с помощью генератора Python.

# Класс Text8Sentences порождает предложения по maxlen слов в каждом из файла text8. В данном случае мы таки читаем весь файл в память,
# но при обходе файлов, находящихся в нескольких каталогах, генератор позволяет загрузить в память часть данных, обработать ее и отдать
# вызывающей стороне:
class Text8Sentences(object):
    def __init__(self, fname, maxlen):
        self.fname = fname
        self.maxlen = maxlen

    def __iter__(self):
        with open(os.path.join(DATA_DIR, "text8"), "rb") as ftext:
            text = ftext.read().split(" ")
            sentences, words = [], []
            for word in text:
                if len(words) >= self.maxlen:
                    yield words
                    words = []
                    words.append(word)
                    yield words

import gensim.downloader as api
import gensim

# Теперь займемся вызывающей программой. В библиотеке gensim используется имеющийся в Python механизм протоколирования для уведомления
# о ходе обработке, поэтому для начала активируем его. В следующей строке создается экземпляр класса Text8Sentences,
# а затем модель обучается на предложениях из на- бора данных. Мы задали размер векторов погружения 300 и рассматриваем
# только слова, которые встречаются в корпусе не менее 30 раз.
# Размер окна по умолчанию равен 5, поэтому контекстом для слова wi будут слова wi­5, wi­4, wi­3, wi­2, wi­1, wi+1, wi+2, wi+3, wi+4, wi+5.
# По умолчанию создается модель CBOW, но это можно изменить, задать параметр sg=1:
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
DATA_DIR = "/Users/grigorev-ee/Work/AnacondaProjects/Idea/Data/"
sentences = Text8Sentences(os.path.join(DATA_DIR, "text8"), 50)
# sentences = api.load('text8')
# model = w2v.Word2Vec(min_count=1)
# model.build_vocab(sentences)

model = gensim.models.Word2Vec(sentences, size=300, min_count=30)

word2vec_load=gensim.models.KeyedVectors.load_word2vec_format(fname='/Users/grigorev-ee/Work/AnacondaProjects/Idea/Data/text8',binary=True)
