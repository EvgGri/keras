# Оценка релевантности слова методом tf-idf (term-frequency - inverse document frequency)
# Когда мы анализируем текстовые данные, мы часто встречаемся со словами в обоих классах (положительный отзыв и отрицательный)
# в двух и более документах. Такие часто встречающиеся слова, как правило, не содержат полезной или отличительной информации.
# tf-idf - частота термина, обратная частота документа, метод позволяет использоваться для понижающего взвешивания частот
# часто встречающихся слов в векторах признаков. (определяется, как произведение этих 2-х показателей)

# tf-idf(t,d)=tf(t,d) x idf(t,d)
# При этом tf(t,d)-частота термина, при этом idf(t,d) = log (n_d / (1+df(d,t))), где n_d - общее число документов,
# df(t,d) - число документов d, которые содержат термин t.
# При этом добавление 1 в знаменатель служит для назначения ненулевого значения терминам, которые встречаются во всех
# тренировочных образцах, логарифм используется для того, чтобы низкие частоты документов гарантированно не получали большие веса.

# -=-=-=-=-=-= Прочитаем необходимые данные

# Модель мешка слов на основе частотности слов
# Метод fit_transform создает словарь и преобразовывает 3 элемента в разреженные векторы признаков
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array(['The sun is shining', 'The weather is sweet', 'The sun is shining and the weather is sweet, and one and one is too'])
bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())

# -=-=-=-=-=-= Рассмотрим реализацию tf-idf в библиотеке scikit-learn -=-=-=-=-=-=-=

# Существует класс-преобразователь коэффициентов tf-idf, который в качестве входных данных принимает из векторизатора частотностей
# CountVectorizer исходные частоты терминов и преобразовывает их в серию tf-idf'ов:
from sklearn.feature_extraction.text import TfidfTransformer
tfidf=TfidfTransformer()
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

# Как мы видели ранее, слово is имело наибольшую частоту термина в 3-ем документе, при этом оно было наиболее встречающимся словом.
# Однако, после преобразования того же вектора признаков в tf-idf мы видим, что в документе 3 слову is теперь поставлен в соответствие
# относительно малый tf-idf(0.39), поскольку оно также содержится в документах 1 и 2 и поэтому в ряд ли будет содержать какую-то полезную
# отличительную информацию.

# В scikit-learn вычисление tf-idf(t,d)=tf(t,d) x (idf(t,d) - 1)
# Перед вычислением tf-idf обычно происходит нормализация исходных частот документов, в scikit-learn по умолчанию используется
# L2-регуляризация, которая вектор признаков делит на его L2-норму, возращая вектор длиной 1.

# -=-=-=-=-=-=-= Очистка текстовых данных -=-=-=-=-=-=-=

# Прочитаем базы отзывов о кинофильмах
import numpy as np
import pandas as pd
df=pd.read_csv('./data/movie_data.csv')

df.loc[23,'review'][:50]
# В отзыве содержится лишняя информация, удалим все лишние за исключением символов-эмоций (эмограммы) вроде ':)'
# Для этого воспользуемся библиотекой регулярных выражений Python re
import re
def preprocessor(text):
    text=re.sub('<[^>]*>','', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-','')
    return(text)
# <[^>]*>' - в первом регулярном выражении мы попытались убрать всю html-разметку из текста
# после этого мы ищем эмограммы, которые мы временно сохранили, как emoticons
# Затем регулярным выражением [\W+] мы удалили из текста все несловарные символы, преобразовали текст в строчные буквы и добавили временно
# сохраненные emoticons в конец обработанной последовательности символов документа, кроме того, мы удалили из эмограмм символ носа '-'

preprocessor(df.loc[23,'review'][:50])
df['review'] = df['review'].apply(preprocessor)

# -=-=-=-=-=-=-= Переработка документов в лексеммы -=-=-=-=-=-=-=
# Как разделить текстовый корпус на отдельные элементы? - Один из способов, разделить на слова по пробельым символам.
def tokenizer(text):
    return text.split()

tokenizer("Example of the splitting")
# В контексте лексемизации другой метод состоит в выделении основы слова (word stemming) - процесс редукции слова к его основе.
# Оригинальный алгоритм стемминга был разработан Мартином Портером, он используется в библиотеке NLTK
# Более новый алгоритм-стеммер Snowball (eng), стоит отметить, что данные алгоритмы могут создавать и несуществующие слова.
# Но алгоритмы лемматизации стремятся получить канонические формы отдельных слов - леммы.
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

tokenizer_porter("Example of the splitting")

# -=-=-=-=-=-=-= Удаление стоп-слов -=-=-=-=-=-=-=
# Стоп-слова: распространенные слова во всех видах текстов, которые несут в себе мало полезной информации.
# Примеры стоп-слов: is, had, and.
# В случае, если мы работаем с исходными или нормализованными частотами терминов, а не tf-idf'ами, которые и без того
# понижают веса часто встречающихся слов, бывает полезно удалить стоп-слова.
# Воспользуемся набором из 127 анлийских стоп-слов, имеющихся в библиотеке NLTK
import nltk
nltk.download('stopwords')
# nltk.download('all')
# Загрузим лингвистический корпус Университета Брауна
from nltk.corpus import brown
brown.words()

from nltk.corpus import stopwords
stop = stopwords.words('english')
[w  for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]

# -=-=-=-=-=-=-= Тренировка лонистической регрессионной модели для задачи классификации документов -=-=-=-=-=-=-=
# Классификация по 2-м классам: положительные отзывы и отрицательные отзывы
X_train=df.loc[:25000,'review'].values
y_train=df.loc[:25000,'sentiment'].values
X_test=df.loc[25000:,'review'].values
y_test=df.loc[25000:,'sentiment'].values

# Затем воспользуемся объектом сеточного поиска GridSearchCV, чтобы определить оптимальные параметры для модели на 5-ти блочной
# стратифицированной перекрестной проверке
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Инициализируем объект GridSearchCV и его сетку параметров
# В приведенном ниже примере мы заменили векторизатор частотностей CountVectorizer и преобразователь tf-idf'ов TfidfTransformer
# на векторизатор TfidfVectorizer, который объединяет данные объекты.
# Наша сетка параметров param_grid состояла из 2-х словарей параметров, в первом словаре мы использовали векторизатор TfidfVectorizer
# с его настройками по умолчанию, чтобы вычислить tf-idf'ы, во втором словаре мы установили эти параметры use_idf=False,
# smooth_idf=False, norm = None, чтобы натренировать модель на основе исходных частот терминов.
# Кроме того, для самой модели логистического классификатора, мы натренировали модели с использованием l1 & l2 регуляризации и штрафного
# параметра clf__penalty и сравнили разные силы регуляризации, задав параметр обратной регуляризации C.
tfidf=TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
param_grid = [{'vect__ngram_range': [(1,1)],
               'vect__stop_words': [stop,None],
               'vect__tokenizer': [tokenizer,tokenizer_porter],
               'clf__penalty': ['l1','l2'],
               'clf__C': [1.0,10.0,100.0]},
               {
               'vect__ngram_range': [(1,1)],
               'vect__stop_words': [stop,None],
               'vect__tokenizer': [tokenizer,tokenizer_porter],
               'vect__use_idf': [False],
               'vect__norm': [None],
               'clf__penalty': ['l1','l2'],
               'clf__C': [1.0,10.0,100.0]
               }
               ]

lr_tfidf=Pipeline([('vect', tfidf),
                   ('clf',LogisticRegression(random_state=0))
                  ])
gs_lr_tfidf=GridSearchCV(lr_tfidf, param_grid, scoring='accuracy',cv=5, verbose=1, n_jobs=-1)
gs_lr_tfidf.fit(X_train,y_train)
print('Наилучший набор параметров: %s' % gs_lr_tfidf.best_params_)
# Мы получили наилучшие результаты поиска по сетке параметров с применением обычного анализатора лексем стемминга Портера,
# без библиотеки стоп-слов и tf-idf'ов в сочетании с логистическим регрессионным классификатором, который использует регуляризацию L2
# с силой регуляризации C=10.0
# На основе наилучшей модели, полученной в результате поиска по сетке параметров, распечатаем среднюю оценку верности по результатам
# 5-ти блочной перекрестной проверки на тренировочном наборе и верность классифкатора на тестовом наборе.
print('Перекрестно-проверочная верность: %.3f' % gs_lr_tfidf.best_score_)
clf=gs_lr_tfidf.best_estimator_
print('Верность на тестовом наборе: %.3f' % clf.score(X_test,y_test))
