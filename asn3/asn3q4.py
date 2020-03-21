import nltk
import math
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
from gensim import corpora
from gensim.models import TfidfModel
from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix, MatrixSimilarity

import os.path


def main():
    tfidf = None
    word2vec = None
    similarityMatrix = None
    browndict = {}
    corporadict = None

    word2vec = None

    choice = ""
    while choice != "exit":
        choice = ""
        while choice not in ["tfidf", "word2vec", "exit"]:
            choice = input("TF-IDF or Word2Vec? [TFIDF, Word2Vec, Exit]\n>").lower()
        
        if choice == "exit":
            break        
        
        catType = ""
        while catType not in ["within", "between", "return"]:
            catType = input("Within or between clusters? [Within, Between, Return]\n>").lower()
        
        if catType == "return":
            break

        # get all of the words for each document per category
        texts = []
        if catType == "within":
            for c in brown.categories():
                words = NormalizeWords(brown.words(categories=c))
                texts.append(words)
                # build a dictionary for me to use later
                browndict[c] = words
        elif catType == "between":
            for c in brown.categories():
                words = NormalizeWords(brown.words(categories=c))
                texts.append(words[:len(words)//2])
                texts.append(words[len(words)//2:])
                # build a dictionary for me to use later
                browndict[c+"1"] = words[:len(words)//2]
                browndict[c+"2"] = words[len(words)//2:]


        # create the corpora dictionary built from gensim
        corporadict = corpora.Dictionary(texts)
        # create a corpus for the training
        corpus = []
        for line in texts:
            corpus.append(corporadict.doc2bow(line))


        if choice == "tfidf":
            # create the tfidf model from our built corpus
            tfidf = TfidfModel(corpus=corpus)

            # build the similarity matrix
            similarityMatrix = MatrixSimilarity(corpus, num_features=len(corporadict))
        elif choice == "word2vec":
            word2vec = Word2Vec(brown.sents())

            # build term similiarity matrix from our models word-vector
            termSimilarityIndex = WordEmbeddingSimilarityIndex(word2vec.wv)

            # build sparse similarity matrix
            sparseSimiliarityMatrix = SparseTermSimilarityMatrix(termSimilarityIndex, corporadict)

            # build similarity word-vector
            WV_SimilarityMatrix = SoftCosineSimilarity(corpus, sparseSimiliarityMatrix)

        maxes = {}
        if choice == "tfidf":
            # Print out the code
            keys = list(browndict.keys())
            for i in range(len(keys)):
                query_bow = corporadict.doc2bow(browndict[keys[i]])
                query_tfidf = tfidf[query_bow]

                # Get the similarity of every cluster
                query_similarity = similarityMatrix[query_tfidf]
                mx = 0
                j = 0
                for j in range(len(query_similarity)):
                    if i == j:
                        continue
                    
                    sim = query_similarity[j]

                    # find the maximum that is not itself
                    if sim > mx:
                        mx = sim
                        s = keys[i] + " and " + keys[j]
                    print(keys[i], "and", keys[j], "have a similarity of:",sim)
                maxes[s] = mx
                i += 1
        elif choice == "word2vec":
            # Print out the code
            keys = list(browndict.keys())
            for i in range(len(keys)):
                query_bow = corporadict.doc2bow(browndict[keys[i]])

                # Get the similarity of every cluster
                query_similarity = WV_SimilarityMatrix[query_bow]
                mx = 0
                j = 0
                for j in range(len(query_similarity)):
                    if i == j:
                        continue
                    
                    sim = query_similarity[j]
                    
                    # find the maximum that is not itself
                    if sim > mx:
                        mx = sim
                        s = keys[i] + " and " + keys[j]
                    print(keys[i], "is", keys[j], "have a similarity of:",sim)
                maxes[s] = mx
                i += 1
        
        for key in maxes:
            print("Max similarity for", key, "=", maxes[key])

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