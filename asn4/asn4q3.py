import nltk
import math
from nltk.corpus import brown, stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import gensim
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import random
import numpy as np
import os

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

    vec_size = 100
    i = 500
    print("Training Word2Vec...")

    fil = "w2v_s" + str(vec_size) + "_i_" + str(i) + ".model"
    if os.path.exists(fil):   
        word2vec = Word2Vec.load(fil)
    else:
        word2vec = Word2Vec(docs_vec, size=vec_size, iter=i, workers=4)
        word2vec.save(fil)
    
    print("Generating acceptable input for models...")
    w2v_docs = []
    w2v_cats = []
    for d, c in docs_cat:
        mean = []
        for w in d:
            if w in word2vec.wv.vocab:
                mean.append(word2vec.wv.get_vector(w))
            else:
                mean.append(0)
        mean = np.array(mean).mean(axis=0)
        w2v_docs.append(mean)
        w2v_cats.append(c)

    train_X, test_X, train_y, test_y = train_test_split(w2v_docs, w2v_cats, test_size=.30, random_state=42)

    print("Training the Logistic Regression Model...")
    lg = LogisticRegression(max_iter=10000)
    lg.fit(train_X, train_y)
    print("Finished training...")
    score = lg.score(test_X, test_y) * 100
    print("Accuracy score for Logisitic Regression:", "%.2f" % score, "%")

    print("Training the Neural Network Model...")
    clf = MLPClassifier(hidden_layer_sizes=(3, 3, 3, 3, 3), activation="relu", solver="adam", max_iter=100000, learning_rate_init=0.001, n_iter_no_change=100)
    clf.fit(train_X, train_y)
    print("Finished training...")
    score = clf.score(test_X, test_y) * 100
    print("Accuracy score for the Neural Network:", "%.2f" % score, "%")
    

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