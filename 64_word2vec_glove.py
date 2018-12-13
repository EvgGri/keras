# -=-=-=-=-=-=- Настройка погружений на основе предобученной модели GloVe

# Погружения на основе предобученной модели GloVe настраиваются примерно так же, как в случае модели word2vec.
# На самом деле отличается только код построения матрицы весов для слоя погружения. Только его мы и рассмотрим.

# Есть несколько видов предобученных моделей GloVe. Мы будем работать с той, что обучена на 6 миллиардах лексем и на корпусе
# текстов объемом порядка миллиарда слов из англоязычной википедии. Размер словаря модели составляет примерно 400 000 слов,
# имеются загружаемые файлы для размерности погружения 50, 100, 200 и 300. Мы возьмем файл для размерности 300.

# Единственное, что нужно изменить в коде предыдущего примера, – часть, где создается модель word2vec и инициализируется ее матрица весов.
# А если бы мы взяли модель с размерностью, отличной от 300, то нужно было бы еще изменить константу EMBED_SIZE.

# Векторы записаны в файле в текстовом формате через пробел, поэтому наша первая задача – прочитать их в словарь word2emb.
# Это делается аналогично разбору строки файла данных для модели word2vec.

# import numpy as np
# import collections
#
# GLOVE_MODEL = "/Users/grigorev-ee/Work/AnacondaProjects/Idea/Data/glove.6B.300d.txt"
# word2emb = {}
# fglove = open(GLOVE_MODEL, "rb")
# for line in fglove:
#     cols = line.strip().split()
#     word = cols[0]
#     embedding = np.array(cols[1:], dtype="float32")
#     word2emb[word] = embedding
# fglove.close()
#
# VOCAB_SIZE = 5000
# EMBED_SIZE = 300
# NUM_FILTERS = 256
# NUM_WORDS = 3
# BATCH_SIZE = 64
# NUM_EPOCHS = 10
#
# word2index = collections.defaultdict(int)
#
#
# # Затем создаем матрицу весов погружения размера vocab_sz × EMBED_SIZE и заполняем ее векторами из словаря word2emb.
# # Векторы, которые соответствуют словам, имеющимся в словаре, но отсутствующим в модели GloVe, остаются нулевыми:
# embedding_weights = np.zeros((vocab_sz, EMBED_SIZE))
# for word, index in word2index.items():
#     try:
#         embedding_weights[index, :] = word2emb[word]
#     except KeyError:
#         pass
#
# from gensim.scripts.glove2word2vec import glove2word2vec
# glove2word2vec(glove_input_file=file, word2vec_output_file="gensim_glove_vectors.txt")
# from gensim.models.keyedvectors import KeyedVectors
# model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)



# -=-=-=-=-=

# from numpy import array
# from numpy import asarray
# from numpy import zeros
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers import Embedding
# # define documents
# docs = ['Well done!',
# 		'Good work',
# 		'Great effort',
# 		'nice work',
# 		'Excellent!',
# 		'Weak',
# 		'Poor effort!',
# 		'not good',
# 		'poor work',
# 		'Could have done better.']
# # define class labels
# labels = array([1,1,1,1,1,0,0,0,0,0])
# # prepare tokenizer
# t = Tokenizer()
# t.fit_on_texts(docs)
# vocab_size = len(t.word_index) + 1
# # integer encode the documents
# encoded_docs = t.texts_to_sequences(docs)
# print(encoded_docs)
# # pad documents to a max length of 4 words
# max_length = 4
# padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# print(padded_docs)
# # load the whole embedding into memory
# embeddings_index = dict()
# f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), 'r', encoding='utf-8')
# for line in f:
# 	values = line.split()
# 	word = values[0]
# 	coefs = asarray(values[1:], dtype='float32')
# 	embeddings_index[word] = coefs
# f.close()
# print('Loaded %s word vectors.' % len(embeddings_index))
# # create a weight matrix for words in training docs
# embedding_matrix = zeros((vocab_size, 100))
# for word, i in t.word_index.items():
# 	embedding_vector = embeddings_index.get(word)
# 	if embedding_vector is not None:
# 		embedding_matrix[i] = embedding_vector
# # define model
# model = Sequential()
# e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)
# model.add(e)
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
# # compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# # summarize the model
# print(model.summary())
# # fit the model
# model.fit(padded_docs, labels, epochs=50, verbose=0)
# # evaluate the model
# loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
# print('Accuracy: %f' % (accuracy*100))



# -=-=-=-=-=-=-=-=-=-=-

'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classification of newsgroup messages into 20 different categories).
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)
20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant


BASE_DIR = '/Users/grigorev-ee/Work/AnacondaProjects/Idea/'
GLOVE_DIR = os.path.join(BASE_DIR, 'Data/')
TEXT_DATA_DIR = os.path.join(GLOVE_DIR, '20-newsgroups/')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                with open(fpath, **args) as f:
                    t = f.read()
                    i = t.find('\n\n')  # skip header
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                labels.append(label_id)

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))
