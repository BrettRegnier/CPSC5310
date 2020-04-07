import nltk
import math
from nltk.corpus import brown, stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import gensim
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import random
import numpy as np

def main():
    word2vec = None
    perceptron = None
    model = None

    # prepare the data
    print("Normalizing words from each document given their categories...")
    docs_vec = []
    docs_cat = []
    for c in brown.categories():
        for _id in brown.fileids(categories=c):
            doc = NormalizeWords(brown.words(fileids=_id))
            docs_vec.append(doc)
            docs_cat.append((doc, c))

    # for c in brown.categories():
    #     doc = NormalizeWords(brown.words(categories=c))
    #     docs_vec.append(doc)
    #     docs_cat.append((doc, c))

    # TODO add epochs and size changes to w2v
    print("Training Word2Vec...")
    word2vec = Word2Vec(docs_vec, size=10, iter=100, workers=4)
    word2vec.save("w2v.model")
    word2vec = Word2Vec.load("w2v.model")
    
    print("Generating acceptable input for models...")
    w2v_docs_cats = []
    w2v_docs = []
    w2v_cats = []
    for d, c in docs_cat:
        mean = []
        for w in d:
            if w in word2vec.wv.vocab:
                mean.append(word2vec.wv.get_vector(w))
            else:
                mean.append(0)
        mean = np.array(mean)
        w2v_docs_cats.append((mean.mean(axis=0), c))
    
    print("Shuffling and seperating the data...")
    random.shuffle(w2v_docs_cats)

    for d, c in w2v_docs_cats:
        w2v_docs.append(d)
        w2v_cats.append(c)

    siz = int(len(w2v_docs) * .70)
    train_X, train_y = w2v_docs[:siz], w2v_cats[:siz]
    test_X, test_y = w2v_docs[siz:], w2v_cats[siz:]

    print("Training the Logistic Regression Model...")
    # lg = LogisticRegression(max_iter=5000)
    print("Finished training...")
    # lg.fit(train_X, train_y)
    # print(lg.score(test_X, test_y))

    print("Training the Neural Network Model...")
    clf = MLPClassifier(hidden_layer_sizes=(3, 3, 3, 3, 3), activation="logistic", solver="adam", max_iter=100000, learning_rate_init=0.1, learning_rate="adaptive", verbose=True, n_iter_no_change=100)
    clf.fit(train_X, train_y)
    print("Finished training...")
    print(clf.score(test_X, test_y))
    

def NormalizeWords(words):
    tokens = []
    sw = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    # filter each word
    for w in words:
        w = lemmatizer.lemmatize(w)
        if (w not in sw) and (w.isalnum()):
            tokens.append(w)

    return tokens

if __name__ == "__main__":
    main()