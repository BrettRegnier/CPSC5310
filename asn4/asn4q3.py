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
    # prepare the data
    print("Normalizing words from each document given their categories...")

    docw2v = None
    boww2v = None

    docs_vec = []
    docs_cat_mat = []

    bow_vec = []
    bow_cat_mat = []
    for c in brown.categories():
        for _id in brown.fileids(categories=c):
            doc = NormalizeWords(brown.words(fileids=_id))
            docs_vec.append(doc)
            docs_cat_mat.append((doc, c))

        words = NormalizeWords(brown.words(categories=c))
        bow_vec.append(doc)
        bow_cat_mat.append((doc, c))

    vec_size = 100
    i = 100
    print("Training Document Word2Vec Model...")

    fil = "doc_w2v_s" + str(vec_size) + "_i_" + str(i) + ".model"
    if os.path.exists(fil):   
        docw2v = Word2Vec.load(fil)
    else:
        docw2v = Word2Vec(docs_vec, size=vec_size, iter=i, workers=4)
        docw2v.save(fil)   
    
    print("Training BOW Word2Vec Model...")
    fil = "bow_w2v_s" + str(vec_size) + "_i_" + str(i) + ".model"
    if os.path.exists(fil):   
        boww2v = Word2Vec.load(fil)
    else:
        boww2v = Word2Vec(bow_vec, size=vec_size, iter=i, workers=4)
        boww2v.save(fil)
    
    print("Generating w2v input for models...")
    w2v_docs = []
    w2v_cats = []
    # this block checks if the word in a document is in the word2vec model and if it is get the w2v vector for the word
    # otherwise just fill in zeros for that word.
    for d, c in docs_cat_mat:
        mean = []
        for w in d:
            if w in docw2v.wv.vocab:
                mean.append(docw2v.wv.get_vector(w))
            else:
                mean.append(np.zeros(100))
        mean = np.array(mean).mean(axis=0)
        w2v_docs.append(mean)
        w2v_cats.append(c)

    # split the documents into training data and test data.
    train_X, test_X, train_y, test_y = train_test_split(w2v_docs, w2v_cats, test_size=.30)

    # BOW method
    tr_x = []
    tr_y = []
    te_x = []
    te_y = []
    for d, c in bow_cat_mat:
        
        tr_data, te_data = train_test_split(d, test_size=.3)
        mean = []
        for w in tr_data:
            if w in boww2v.wv.vocab:
                mean.append(boww2v.wv.get_vector(w))
            else:
                mean.append(np.zeros(vec_size))
        tr_x.append(np.array(mean).mean(axis=0))
        tr_y.append(c)

        mean = []
        for w in te_data:
            if w in boww2v.wv.vocab:
                mean.append(boww2v.wv.get_vector(w))
            else:
                mean.append(np.zeros(vec_size))
        te_x.append(np.array(mean).mean(axis=0))
        te_y.append(c)

    print("")
    print("Training the Logistic Regression Model on Documents...")
    lg = LogisticRegression(max_iter=5000, C=1e5)
    lg.fit(train_X, train_y)
    score = lg.score(test_X, test_y) * 100
    print("Accuracy score for Logisitic Regression on Documents:", "%.2f" % score, "%")

    # The predictions print quite long, so I just gave a score.
    # y_pred = lg.predict(test_X)
    # print("Predictions: " + str(list(y_pred)))
    # print("Ground Truth: " + str(list(test_y)))

    print("")
    print("Training the Neural Network Model on Documents...")
    clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", max_iter=100000, learning_rate_init=0.001, warm_start=True)
    clf.fit(train_X, train_y)
    score = clf.score(test_X, test_y) * 100
    print("Accuracy score for the Neural Network on Documents:", "%.2f" % score, "%")

    print("")
    print("Training the Neural Network Model on Documents with small NN...")
    clf = MLPClassifier(hidden_layer_sizes=(3,3,3,3,3), activation="relu", solver="adam", max_iter=100000, learning_rate_init=0.001, warm_start=True)
    clf.fit(train_X, train_y)
    score = clf.score(test_X, test_y) * 100
    print("Accuracy score for the Neural Network on Documents:", "%.2f" % score, "%")
    
    # The predictions print quite long, so I just gave a score.
    # y_pred = clf.predict(test_X)
    # print("Predictions: " + str(list(y_pred)))
    # print("Ground Truth: " + str(list(test_y)))

    print("")
    print("#########################################")
    print("")
    print("Training the Logistic Regression Model on BOW...")
    lg = LogisticRegression(max_iter=5000, C=1e5)
    lg.fit(tr_x, tr_y)
    score = lg.score(te_x, te_y) * 100
    print("Accuracy score for Logisitic Regression on BOW:", "%.2f" % score, "%")

    y_pred = lg.predict(te_x)
    print("Predictions: " + str(list(y_pred)))
    print("Ground Truth: " + str(list(te_y)))

    print("")
    print("Training the Neural Network Model on BOW...")
    clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", max_iter=100000, learning_rate_init=0.001, warm_start=True)
    clf.fit(tr_x, tr_y)
    score = clf.score(te_x, te_y) * 100
    print("Accuracy score for the Neural Network on BOW:", "%.2f" % score, "%")
    
    y_pred = clf.predict(te_x)
    print("Predictions: " + str(list(y_pred)))
    print("Ground Truth: " + str(list(te_y)))
    
    print("")
    print("Training the Neural Network Model on BOW with small NN...")
    clf = MLPClassifier(hidden_layer_sizes=(3,3,3,3,3), activation="relu", solver="adam", max_iter=100000, learning_rate_init=0.001, warm_start=True)
    clf.fit(tr_x, tr_y)
    score = clf.score(te_x, te_y) * 100
    print("Accuracy score for the Neural Network on BOW:", "%.2f" % score, "%")
    
    y_pred = clf.predict(te_x)
    print("Predictions: " + str(list(y_pred)))
    print("Ground Truth: " + str(list(te_y)))

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