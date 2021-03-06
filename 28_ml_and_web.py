# Зачастую придится работать с крупными наборами данных, которые могут превышать память используемого компьютера.
# Для этого рассмотрим обучение вне ядра (out of core learning), т.е. с использованием внешней памяти.
# Воспользуемся функцией partial_fit классификатора на основе стохастического градиентного спуска SGDClassifierself,
# чтобы передавать поток документов непосредственно из нашего локального диска и тренировать логистическую регрессионную
# модель с использованием небольших мини-пакетов документов.

# Определим функцию лексемизации tokenizer, которая очищает необработанные текстовые данные.

# Прочитаем базы отзывов о кинофильмах
import numpy as np
from nltk.corpus import stopwords
import pandas as pd
df=pd.read_csv('./data/movie_data.csv')
stop = stopwords.words('english')

# В отзыве содержится лишняя информация, удалим все лишние за исключением символов-эмоций (эмограммы) вроде ':)'
# Для этого воспользуемся библиотекой регулярных выражений Python re
import re
def tokenizer(text):
    text=re.sub('<[^>]*>','', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-','')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# Определим генераторную функцию stream_docs, которая считывает и выдает по одному документу за раз
def stream_docs(path):
    with open(path,'r',encoding='utf-8') as csv:
        next(csv) # пропускаем заголовок
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

# Первый документ должен вернуть кортеж, состоящий из текста отзыва и соответствующей метки класса
next(stream_docs(path='./data/movie_data.csv'))
# Определем функцию get_minibatch, которая принимает поток документов из функции stream_docs и возвращает отдельно взятое число докуметов,
# заданных в параметре size
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)

    except StopIteration:
            return None, None
    return docs, y

# Мы не можем использовать векторизатор частотностей CountVectorizer для обучения вне ядря, т.к. он потребует наличия в памяти полного словаря.
# Кроме того, векторизатор tf-idf'ов TfidfVectorizer должен поддерживать в памяти все векторы признаков тренировочного набора данных, для того
# чтобы вычислить обратные частоты документов.
# Но существует еще один векторизатор для обработки текст HashingVectorizer, он независим от данных и хэширует признаки на основе 32-рязрядного
# алогритма MurmushHash
# Инициализируем хеширующий векторизатор HashingVectorizer нашей функцией tokenizer и определим число признаков равным 2^21.
# Мы повторно инициализировали логистический регрессионный классификатор, задав параметр потерь loss классификатора SGDClassifier равным log.
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
vect = HashingVectorizer(decode_error = 'ignore',
                         n_features=(2 ** 21),
                         preprocessor=None,
                         tokenizer=tokenizer)

clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='./data/movie_data.csv')

# Настоив вспомогательные функции, приступим к обучению вне ядра.
# PyPrind - следим за ходом выполнения обучения. Мы инициализировали индикатор выполнения работы 45 итерациями, в следующем
# цикле for мы выполнили терации по 45 мини-пакетам документов, где каждый мини-пакет состоит из 1000 документов.
import pyprind
pbar=pyprind.ProgBar(45)
classes = np.array([0,1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size = 1000)
    if not X_train:
        break
    X_train=vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

# Проверим качество моделирования на последних 5000 документах
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test=vect.transform(X_test)
print('Верность: %.3f' % clf.score(X_test, y_test))

# Верность модели составляет 87%, слегка ниже верности, которую мы достигли в предыдущем разделе в результате поиска по сетке параметров.
# Вместе с тем обучение вне ядра очень эффективно использует памятьит заняло менее минуты.

# В заключение мы можем использовать последние 5000 документов для обновления нашей модели.
clf=clf.partial_fit(X_test, y_test)
# Несмотря на то, что модель мешка слов является наиболее распространенной моделью классификации текстов, она не рассматривает структуру
# предложений и грамматику. Популярным расширением модели мешка слов является тематическая модель латентного распределения Дирихле,
# которая рассматривает скрытую семантику слов.
# Алгоритм word2vec - это алгоритм обучения без учителя на основе нейронных сетей, который автоматически пытается получить взаимосвязь
# между словами.
# В основе алгоритма word2vec лежит идея, которая состоит в том, чтобы помещать слова, имеющие подобные значения в подобные группы, благодаря
# умному распределению векторного пространства (vector-spacing) модель может воспроизводить определенные слова при помощи простой векторной
# математике, например: король-мужчина + женщина = королева

# -=-=-=-=-=-=-=-=-=-=- Встраивание в модели в web-приложение -=-=-=-=-=-=-=-=-=-=-=-=-=-

# Разумеется, мы не тренируем модель заново всякий раз, когда закрываем интерпретатор Python.
# Одним из вариантов обеспечения перманентности модели является модуль косервации данных pickle, который позволяет делать сериализацию и
# десериализацию объектных структур Python в виде компактного байт-кода, в результате чего мы сможеми сохранять классификатор в его
# текущем состоянии и перезагружать его, если мы хотим выполнить классификацию новых образцов без необходимости снова и снова
# извлекать модель из тренировочных данных.
import pickle
import os

dest = os.path.join('movieclassifier', 'pkl_objects')

# Создадим каталог классификатора movieclassifier, куда мы позже будем охранять файлы и данные для нашего веб-приложения.
# В каталоге movieclassifier создаем подкаталог pkl_objects для сохранения на нашем локальном диске сериализованных объектов Python
# При помощи метода dump мы потом сериализовали натренированную логистическую регриссионную модель, а также набор стоп-слов из библиотеки
# NLTK. Метод dump в качестве своего первого аргумента принимает объект, который мы хотим законсервировать, и в качестве второго
# аргумента мы предоставили открытый файловый объект, куда бъект Python будет записан. При помощи аргумента wb в функции open, мы открыли
# файл в двоичном режиме для выполнения консервации и задали параметр протокола protocol=4, чтобы выбрать самый последний и эффективный
# протокол консервации.
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'),'wb'), protocol=4)
pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'),'wb'), protocol=4)

# Нам не нужно консервировать хэширующий векторизатор HashingVectorizer, поскольку он не требует выполнения подгонки. Вместо этого можно
# создать новый сценарный файл Python, из которого можно импортировать векторизатор в наш текущий сеанс Python.

# После того, как мы успешно загрузили векторизатор и расконсервировали классифкатор, теперь можно использовать эти объекты для
# предобработки образцов документов и прогнозировать мнения:
import numpy as np

# С учетом того, что наш классификатор возвращает метки классов в виде целых чисел, мы определили простой словарь Python, в котором
# целым числам в соответствие поставлены мнения.
label = {0: 'negative', 1: 'positive'}

example = ['I love this movie']
# Затем мы применили хэширующий векторизатор HashingVectorizer для преобразования простого примера документа в вектор Х.
X = vect.transform(example)

# В заключении мы применили метод predict логистического регрессионного классификатора для прогнозирования метки класса.
# Метод predict_proba возращает массив вероятностей для всех меток классов, поскольку метка класса с самой большой вероятностью
# соответствует метке класса, полученной из метода predict, мы использовали np.max
print('Прогноз: %s\nВероятность: %.2f%%' % (label[clf.predict(X)[0]], np.max(clf.predict_proba(X))*100))
