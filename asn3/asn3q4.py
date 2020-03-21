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


def main2():
    # TODO normalize the data, and build matrix of docs, with their words, frequencies, and appearances
    choice = ""
    while choice != "exit":
        choice = ""
        while choice not in ["brown", "import", "exit"]:
            choice = input("Find cosine difference for Brown Corpus, or import 2 docs? [Brown, Import, Exit]\n>").lower()
        
        if choice == "exit":
            break


        docs = {}
        appearVec = {}
        if choice == "brown":
            # TODO between clusters by having a prompt to do so.
            between_cluster = []
            full_docs = []
            half_docs = []
            idx = 0
            for c in brown.categories():
                words = brown.words(categories=c)
                docs.append({'doc': c, "freq_vec": NormalizeWords_BuildFreqVector(words)})
                if idx == 2:
                    break
                idx += 1
                # full_cluster.append({"doc": c, "words": brown.words(categories=c)})

                # s = len(brown.words(categories=c))//2
                # half_docs.append({"doc": c, "words": brown.words(categories=c)[:s]})
                # half_docs.append({"doc": c, "words": brown.words(categories=c)[s:]})
                # TODO between "clusters"
                # first half
                # clusters.append(NormalizeWords(words[:len(words)]))
                # seconds half
                # clusters.append(NormalizeWords(len(words):))
        elif choice == "import":
            path1 = "F:/code/py/cpsc5310/asn3/doc1.txt"
            path2 = "F:/code/py/cpsc5310/asn3/doc2.txt"
            doc1 = None
            doc2 = None
            while not os.path.isfile(path1):
                path1 = input("Path to doc 1 [./path/to/file]\n>").lower()
                if path1.lower() == "exit":
                    exit()
            while not os.path.isfile(path2) and path2 != path1:
                path2 = input("Path to doc 2 [./path/to/file]\n>").lower()
                if path2.lower() == "exit":
                    exit()

            n1 = os.path.splitext(os.path.basename(path1))[0]
            n2 = os.path.splitext(os.path.basename(path2))[0]
            
            doc1 = open(path1).read()
            doc2 = open(path2).read()

            doc1 = nltk.word_tokenize(doc1)
            doc2 = nltk.word_tokenize(doc2)

            docs[n1] = NormalizeWords_BuildFreqVector(doc1)
            docs[n2] = NormalizeWords_BuildFreqVector(doc2)


        appearVec = CountAppearInDocs(docs)

        tf_idf = TF_IDF(docs, appearVec)

        similarities = CosineSimilarity(tf_idf)
        print(similarities)

def NormalizeWords_BuildFreqVector(words):
    freqVec = {}
    sw = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    for word in words:
        word = lemmatizer.lemmatize(word)
        if (word not in sw) and (word.isalnum()):
            if word in freqVec:
                freqVec[word] += 1
            else:
                freqVec[word] = 1

    # print(freqVec)
    return freqVec

# takes in a list of docs that will be compared and finds
# how many times the the word appears in the doc for tf-idf
def CountAppearInDocs(docs):
    appearVec = {}
    for doc in docs:
        for word in docs[doc]:
            if word in appearVec:
                appearVec[word] += 1
            else:
                appearVec[word] = 1
    
    return appearVec

def NormalizeAndBuildFreqMatrix(docs):
    # strip out all of the useless stuff.
    freqMatrix = []
    sw = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    docidx = 0
    for doc in docs:
        for word in doc['words']:
            word = lemmatizer.lemmatize(word)
            if (word not in sw) and (word.isalnum()):
                exist = False
                idx = 0
                for item in freqMatrix:
                    if word == item['word']:
                        exist = True
                        break
                    idx += 1
                if not exist:
                    freq = [0]*len(docs)
                    freq[docidx] = 1
                    freqMatrix.append({'word': word, 'freqs': freq})
                else:
                    freqMatrix[idx]['freqs'][docidx] += 1
        docidx += 1

        for item in freqMatrix:
            c = 0
            for freq in item['freqs']:
                if freq > 0:
                    c += 1
            item['appears'] = c

    return freqMatrix

def TF_IDF(docs, appearVec):
    tfidf_matrix = {}
    num_docs = len(docs)
    for doc in docs:
        if doc not in tfidf_matrix:
            tfidf_matrix[doc] = {}
        for word in docs[doc]:
            if word not in tfidf_matrix[doc]:
                for d in docs:
                    if d not in tfidf_matrix:
                        tfidf_matrix[d] = {}
                    tfidf_matrix[d][word] = 0
            
            tf = 0
            idf = 0

            if docs[doc][word] > 0:
                tf = 1 + math.log(docs[doc][word], 10)
            
            df = appearVec[word]
            idf = math.log(num_docs/df, 10)

            tfidf_matrix[doc][word] = tf * idf

    # print(tfidf_matrix)
    return tfidf_matrix

# takes in a dictionary of docs of their TF_IDF value or
# Word2Vec values
def CosineSimilarity(docs):
    s = len(docs)
    similarities = {}
    idx = 1
    for v in docs:
        keys = list(docs.keys())[idx:]
        for w in keys:
            product = 0
            veclen1 = 0
            veclen2 = 0
            similarity = 0
            for word in docs[v]:
                product += docs[v][word] * docs[w][word]
                veclen1 += docs[v][word] * docs[v][word] # pow 2
                veclen2 += docs[w][word] * docs[w][word] # pow 2
            
            if veclen1 > 0 and veclen2 > 0:
                similarity = product / (math.sqrt(veclen1) * math.sqrt(veclen2))
            similarities[v + "_" + w] = similarity

        idx += 1
    return similarities
                
if __name__ == "__main__":
    main()