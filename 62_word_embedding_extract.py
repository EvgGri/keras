# Извлечение погружений word2vec из модели

# Выше уже отмечалось, что хотя обе модели семейства word2vec можно свести к задаче классификации, сама эта задача нас не интересует.
# А интересен нам побочный эффект классификации – матрица весов, которая преобразует слово из словаря в плотное распределенное
# представление низкой размерности.

# Есть немало примеров того, как распределенные представления выявляют удивительную синтаксическую и семантическую информацию.
# Так, на следующем рисунке, взятом из презентации Томаша Миколова на конференции NIPS 2013 (см. T. Mikolov, I. Sutskever, K. Chen,
# G. S. Corrado, J. Dean, Q. Le, T. Strohmann «Learning Representations of Text using Neural Networks», NIPS 2013), показано,
# что векторы, соединяющие слова, имеющие одинаковый смысл, но относящиеся к разным полам, приблизительно параллельны в редуцированном
# двумерном пространстве и что зачастую можно получить согласующиеся с интуицией результаты, производя арифметические действия
# над векторами слов. В презентации много таких примеров.

# Интуитивно кажется, что процесс обучения привносит достаточно информации во внутреннюю кодировку, чтобы предсказать выходное слово,
# встречающееся в контексте входного. Поэтому точки, представляющие слова в этом пространстве, располагаются ближе к точкам слов,
# с которыми они встречаются совместно. В результате похожие слова образуют кластеры. И слова, встречающиеся вместе с похожими словами,
# тоже образуют кластеры. Следовательно, векторы, соединяющие точки, представляющие семантически связанные слова в распределенном
# представлении, демонстрируют регулярное поведение.

# -=-=-=-=-=-=-=-=-=-=-=-=-

# Опишем, как строится модель skip­ грамм в Keras. Предположим, что размер словаря равен 5000, размер выходного пространства погружения 300,
# размер окна 1. Последнее означает, что контекст состоит из предыдущего и следующего слова. Сначала импортируем нужные модули и
# инициализируем переменные:
from keras.layers import merge,Concatenate # аналог keras.layers.merge
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

# -=-=-=-=-=-=-=-=-=-=-=-=-

# Keras предоставляет средства для извлечения весов из обученных моделей.
# В случае skip­грамм веса слоя погружения можно получить следующим образом:
merge_layer = model.layers[0]
word_model = merge_layer.layers[0] # тут модель падает, надо разбираться
word_embed_layer = word_model.layers[0]
weights = word_embed_layer.get_weights()[0]

# А в случае CBOW для получения весов достаточно одной строчки:
weights = model.layers[0].get_weights()[0]

# В обоих случаях матрица весов имеет размер vocab_size × embed_ size. Для вычисления распределенного представления слова из словаря
# нужно построить унитарный вектор, записав 1 в элемент вектора размера vocab_size с индексом, равным идентификатору слова,
# и умножить его на матрицу весов, получив в результате вектор погружения размера embed_size.
