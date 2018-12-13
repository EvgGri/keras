import itertools

from gensim.models.word2vec import Text8Corpus

from glove import Glove
from glove import Corpus

sentences = list(itertools.islice(Text8Corpus('/Users/grigorev-ee/Work/AnacondaProjects/Idea/Data/text8'),None))

corpus = Corpus()

corpus.fit(sentences, window=10)

glove = Glove(no_components=100, learning_rate=0.05)

glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
