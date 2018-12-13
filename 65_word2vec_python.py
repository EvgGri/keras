#=-=-=-=-=-=-=-=-=-=-=- Word2Vec in Python

from gensim.models import word2vec
import logging
import gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.Text8Corpus('/Users/grigorev-ee/Work/AnacondaProjects/Idea/Data/text8')

model = word2vec.Word2Vec(sentences, size=200)

model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
model.most_similar(positive=['woman', 'king'], negative=['man'], topn=2)
model.most_similar(['man'])

model.save('./data/text8.model')
model.wv.save_word2vec_format('./data/text.model.bin', binary=True)


model1 = gensim.models.KeyedVectors.load_word2vec_format('./data/text.model.bin', binary=True)
model1.most_similar(['girl', 'father'], ['boy'], topn=3)

more_examples = ["he is she", "big bigger bad", "going went being"]

for example in more_examples:
    a, b, x = example.split()
    predicted = model.wv.most_similar([x, b], [a])[0][0]

print("'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted))
