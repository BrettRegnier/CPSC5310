import nltk
import math

def main():
    # TODO normalize the data, and build matrix of docs, with their words, frequencies, and appearances
    pass

# docs is an array of documents.
# "word": string -the word that is accociated.
# "freq": int -count of how many times it appears in the doc
# "appears": int -count of how many docs it appears in, 1 or 2
def TF_IDF(docs):
    tfidf_matrix = []
    idx = 0
    for doc in docs:
        tfidf_matrix.append([])
        for word in doc:
            tf = 0
            idf = 0
            if word[1] > 0:
                tf = 1 + math.log(word[1], 10)

            # magic number of 2 because we are comparing two documents always according the the question.
            idf = math.log(2/word[2], 10)

            # add it to the tfidxmatrix
            tfidf_matrix[idx].append({"word": word[0], "tf_idf": tf*idf})

        idx += 1
    return tfidf_matrix

def Word2Vec():
    pass

def CostineSimilarity():
    pass

if __name__ == "__main__":
    main()