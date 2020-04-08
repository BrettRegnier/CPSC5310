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

######################################################################################################
# The goal of this program is to classify texts into categories based on the words of the document. 
# My program takes each category and gets the words from each file given that category and normaizes the data.
# Then I get the w2v vector for each document and place them into a list and split the documents into
# two sets, the training set and the testing set. Each with their correct classified category.
# 
# Problem: Since there is very little documents (only 500) to train on, the accuracy for both the logistic regression
# and Neural Network is quite low due to too little information to train on. 
#
# Solution: A larger dataset would be trained on more effectively, however since we are only using the brown corpus it is difficult
# a potentional solution could be to treat the categories are a bag of words and classify a subset from the bag of words. However,
# in real world problems a entire corpus wouldn't be classified by taking a BOW, but a document from a corpus would be treated as such instead.
#
########################################################################################################

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
    i = 100
    print("Training Word2Vec...")

    fil = "w2v_s" + str(vec_size) + "_i_" + str(i) + ".model"
    if os.path.exists(fil):   
        word2vec = Word2Vec.load(fil)
    else:
        word2vec = Word2Vec(docs_vec, size=vec_size, iter=i, workers=4)
        word2vec.save(fil)
    
    print("Generating w2v input for models...")
    w2v_docs = []
    w2v_cats = []
    # this block checks if the word in a document is in the word2vec model and if it is get the w2v vector for the word
    # otherwise just fill in zeros for that word.
    for d, c in docs_cat:
        mean = []
        for w in d:
            if w in word2vec.wv.vocab:
                mean.append(word2vec.wv.get_vector(w))
            else:
                mean.append(np.zeros(100))
        mean = np.array(mean).mean(axis=0)
        w2v_docs.append(mean)
        w2v_cats.append(c)

    # split the documents into training data and test data.
    train_X, test_X, train_y, test_y = train_test_split(w2v_docs, w2v_cats, test_size=.30, random_state=0)

    print("Training the Logistic Regression Model...")
    lg = LogisticRegression(max_iter=50000, C=1e5)
    lg.fit(train_X, train_y)
    print("Finished training...")
    score = lg.score(test_X, test_y) * 100
    print("Accuracy score for Logisitic Regression:", "%.2f" % score, "%")

    print("Training the Neural Network Model...")
    clf = MLPClassifier(hidden_layer_sizes=(3, 3, 3, 3, 3), activation="relu", solver="adam", max_iter=100000, learning_rate_init=0.001, n_iter_no_change=500)
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
            tokens.append(w.lower())

    return tokens

if __name__ == "__main__":
    main()