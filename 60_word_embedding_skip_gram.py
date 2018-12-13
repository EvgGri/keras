#-=-=-=-=-=-=-=-=- Погружения слов

# В википедии погружение, или векторное представление слов (word embedding) определяется как общее название различных методов языкового
# моделирования и обучения признаков, применяемых в обработке естественных языков (ОЕЯ, англ. NLP), когда слова или фразы из словаря
# отображаются на векторы вещественных чисел.
#
# Погружение слов – это способ преобразовать текстовое представление слов в числовые векторы, допускающие анализ стандартными алгоритмами
# машинного обучения, принимающими на входе числа.
#
# В главе 1 мы уже встречались с одним видом погружения слов – унитарным кодированием. Это самый простой подход к погружению.
# Напомним, что унитарным кодом слова будет вектор, число элементов которого равно размеру словаря, такой, что элемент, соответствующий
# данному слову, равен 1, а все остальные 0.
#
# Основная проблема унитарного кодирования в том, что нет никакого способа представить сходство слов. В любом заданном корпусе текстов
# мы ожидаем, что между словами «кошка» и «собака» или «нож» и «вилка» есть какое­то сходство. Сходство векторов вычисляется с помощью
# скалярного произведения, т. е. суммы произведений соответственных элементов.
#
# В случае унитарного кодирования скалярное произведение любых двух слов равно нулю.
# Для преодоления ограничений унитарного кодирования сообщество ОЕЯ заимствовало из информационного поиска (ИП) идею
# векторизации текста с использованием документа в качестве контекста. Здесь стоит отметить такие подходы,
# как TF­-IDF (https://en.wikipedia.org/wiki/Tf%E2%80%93idf), латентно-семантический анализ (ЛСА)
# (https://en.wikipedia.org/wiki/Latent_semantic_ analysis) и тематическое моделирование (https://en.wikipedia.org/ wiki/Topic_model).
#
# Но эти представления улавливают несколько иную, документо­центрическую, идею семантического сходства.

# Рассмотрим две формы погружения слов, GloVe и word2vec, известные под общим названием «распределенное представление слов».

# Мы также узнаем о способах порождения собственных погружений в программе на Keras, а равно о том, как использовать и настраивать
# предобученные модели на основе word2vec и GloVe.

# Будут рассмотрены следующие темы:
#  построение различных распределенных представлений слов в контексте;
#  построение моделей на основе погружений для решения таких задач ОЕЯ, как грамматический разбор предложения и анализ эмоциональной окраски.

# Распределенные представления
# Распределенное представление – это попытка уловить смысл слова путем рассмотрения его связей с другими словами в контексте.
# Эта идея сформулирована в следующем высказывании Дж. Р. Фир- та (J. R. Firth)
# (см. статью Andrew M. Dai, Christopher Olah, Quoc V. Le «Document Embedding with Paragraph Vectors», arXiv:1507.07998, 2015), лингвиста,
# который первым выдвинул ее:

# Мы узнаем слово по компании, с которой оно дружит.
# Рассмотрим такие два предложения:
# Париж – столица Франции.
# Берлин – столица Германии.

# Даже если вы совсем не знаете географию (или русский язык), все равно нетрудно сообразить, что пары слов (Париж, Берлин) и (Франция, Германия)
# как­то связаны и что между соответственными словами связи одинаковы, т. е.
# Париж : Франция :: Берлин : Германия

# Следовательно, задача распределенного представления – найти такую общую функцию φ преобразования слова в соответствующий ему вектор,
# что справедливы соотношения следующего вида:
# φ («Париж») – φ («Франция») ≈ φ («Берлин») – φ («Германия»)

# Иными словами, цель распределенного представления – преобразовать слова в векторы, так чтобы сходство векторов коррелировало с семантическим
# сходством слов.
#
# В следующих разделах мы рассмотрим два наиболее известных погружения слов: word2vec и GloVe.

#-=-=-=-=-=-=-=-=- word2vec

# Группа моделей word2vec была разработана в 2013 году группой исследователей Google под руководством Томаша Миколова (Tomas Mikolov).
# Модели обучаются без учителя на большом корпусе текстов и порождают векторное пространство слов.
# Размерность пространства погружения word2vec обычно меньше размерности пространства погружения для унитарного кодирования,
# которая равна размеру словаря. Кроме того, это пространство погружения плотнее разреженного погружения при унитарном кодировании.

# Существуют две архитектуры word2vec:
# 1. непрерывный мешок слов (Continuous Bag Of Words, CBOW);
# 2. skip­ граммы.

# В архитектуре CBOW модель предсказывает текущее слово, если известно окно окружающих его слов.
# Кроме того, порядок контекстных слов не влияет на предсказание (это допущение модели мешка слов).
# В архитектуре skip­ грамм модель предсказывает окружающие слова по известному центральному слову.
# Согласно заявлению авторов, CBOW быстрее, но skip­граммы лучше предсказывают редкие слова.

# Интересно отметить, что хотя word2vec создает погружения, используемые в моделях глубокого обучения ОЕЯ, оба варианта word2vec,
# которые мы будем обсуждать и которые снискали наибольший успех и признание, являются мелкими нейронными сетями.

# -=-=-=-=-=- Модель skip-грамм

# Модель skip ­грамм обучается предсказывать окружающие слова по известному центральному. Чтобы понять, как она работает, рассмотрим
# такое предложение:

# I love green eggs and ham. (Я люблю зеленые яйца и ветчину)

# В предположении, что размер окна равен 3, предложение можно
# разложить на такие пары (контекст, слово):
# ([I, green], love)
# ([love, eggs], green)
# ([green, and], eggs)

# Поскольку модель skip­ грамм предсказывает контекстное слово по центральному, мы можем преобразовать этот набор данных в набор пар (вход, выход).
# То есть, зная входное слово, мы ожидаем, что модель skip­грамм предскажет соответствующее выходное:
# (love, I), (love, green), (green, love), (green, eggs), (eggs, green), (eggs, and), ...

# Мы можем также сгенерировать дополнительные отрицательные примеры, объединяя в пару каждое входное слово со случайным словом из словаря,
# например:
# (love, Sam), (love, zebra), (green, thing), ...

# Наконец, мы генерируем положительные и отрицательные примеры для классификатора:
# ((love, I), 1), ((love, green), 1), ..., ((love, Sam), 0), ((love, zebra), 0), ...

# Теперь можно обучить классификатор, который принимает вектор слов и контекстный вектор и предсказывает 0 или 1 в зависимости от того,
# какой пример видит: положительный или отрицательный. Результатом обучения сети являются веса слоя погружения слов.

# Опишем, как строится модель skip­ грамм в Keras. Предположим, что размер словаря равен 5000, размер выходного пространства погружения 300,
# размер окна 1. Последнее означает, что контекст состоит из предыдущего и следующего слова. Сначала импортируем нужные модули и
# инициализируем переменные:
from keras.layers import Concatenate # аналог keras.layers.merge
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Sequential

vocab_size = 5000
embed_size = 300

# Затем создадим последовательную модель слова. Входом модели являются идентификаторы слов в словаре.
# Весам погружений первоначально присваиваются небольшие случайно выбранные значения.
# В процессе обучения модель будет обновлять веса, применяя алгоритм обратного распространения.
# Следующий слой адаптирует форму входа под размер погружения:

word_model = Sequential()
word_model.add(Embedding(vocab_size, embed_size,embeddings_initializer="glorot_uniform",input_length=1))
word_model.add(Reshape((embed_size, )))

# Нам также нужна еще одна модель для контекстных слов. Для каждой пары skip­ грамм мы имеем одно контекстное слово, соответствующее
# целевому слову, так что эта модель идентична модели слов:
context_model = Sequential()
context_model.add(Embedding(vocab_size, embed_size,embeddings_initializer="glorot_uniform",input_length=1))
context_model.add(Reshape((embed_size,)))

# На выходе обеих моделей получаются векторы размера embed_size. Их скалярное произведение подается на вход плотному слою с
# сигмоидной функцией активации, который порождает один выход. Напомним, что сигмоида преобразует свой аргумент в число из
# диапазона [0,1] и асимптотически быстро приближается к единице на положительной полуоси и к 0 – на отрицательной.
model = Sequential()
model.add(Concatenate([word_model, context_model]))
model.add(Dense(1, init="glorot_uniform", activation="sigmoid"))
model.compile(loss="mean_squared_error", optimizer="adam")

# В качестве функции потерь используется mean_squared_error; идея в том, чтобы минимизировать скалярное произведение для положительных примеров
# и максимизировать для отрицательных. Напомним, что скалярное произведение равно сумме произведений соответственных элементов,
# поэтому скалярное произведение похожих векторов будет больше, чем непохожих, т. к. у похожих векторов больше одинаковых элементов в
# соответственных позициях.

# Keras предоставляет функцию для извлечения skip­ грамм из текста, преобразованного в список индексов слов.
# Ниже приведен пример ее использования для извлечения первых 10 из 56 сгенерированных skip­грамм (положительных и отрицательных).

# Сначала импортируем пакеты и инициализируем подлежащий анализу текст:
from keras.preprocessing.text import *
from keras.preprocessing.sequence import skipgrams

text = "I love green eggs and ham ."

# Затем объявляем объект для выделения лексем и пропускаем через него текст. Получается список лексем:
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

# Объект tokenizer создает словарь, сопоставляющий каждому уникальному слову целочисленный идентификатор, и делает его доступным через
# атрибут word_index. Мы читаем этот атрибут и создаем две таблицы соответствия:
word2id = tokenizer.word_index
id2word = {v:k for k, v in word2id.items()}

# Наконец, входной список слов преобразуется в список идентификаторов и передается функции skipgrams.
# Затем мы печатаем первые 10 из 56 сгенерированных skip­грамм (пара, метка):
wids = [word2id[w] for w in text_to_word_sequence(text)]
pairs, labels = skipgrams(wids, len(word2id))
print(len(pairs), len(labels))
for i in range(15):
    print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
    id2word[pairs[i][0]], pairs[i][0], id2word[pairs[i][1]], pairs[i][1],
    labels[i]))

# Выше показаны результаты работы этой программы. На вашей машине результаты могут отличаться, потому что функция skipgrams
# производит случайную выборку из множества возможных положительных примеров. Кроме того, генерация отрицательных примеров производится
# путем выборки случайных пар лексем из текста. С увеличением размера входного текста вероятность выбрать пары не связанных
# между собой слов возрастает. Но в нашем примере текст очень короткий, поэтому высоки шансы, что будут сгенерированы и
# положительные примеры тоже.