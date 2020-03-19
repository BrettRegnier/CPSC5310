import nltk
from nltk.corpus import brown
from gensim.models import Word2Vec

model = Word2Vec(brown.sents())

model.save('brown.embedding')
new_model = Word2Vec.load('brown.embedding')

print(new_model.similarity('university', 'school') > 0.3)