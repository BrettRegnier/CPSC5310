from collections import Counter

import nltk
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.lm import Vocabulary
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer


def PrintTagandVocabCount(words):

    # get tags via nltk
    tags = nltk.pos_tag(words)

    # list of all unique tags and their count
    ltags = []  # format [["tag", x], ...]

    # loop through all tags
    for t in tags:
        tag = t[1]  # get the tag from object

        # bool for if the tag should be appended to the list of tags
        unique = True

        # loop through current list of tags
        for li in ltags:
            # tag is not unique
            if li[0] == tag:
                unique = False
                li[1] += 1
                break

        if unique:
            ltags.append([tag, 1])

    # print("tags:", ltags) # debug
    print("unique tag count:", len(ltags))

    vocab = Vocabulary(words)
    # print(vocab.counts) # debug
    print("vocabulary size:", len(vocab.counts))
    print("")


stopwords = stopwords.words('english')

# Note. I loop through the categories multiple times because I want
# to keep the questions seperated.

print("----- Brown details with stopwords -----")
for c in brown.categories():
    words = brown.words(categories=c)
    print("category:", c)
    print("word count:", len(words))

    PrintTagandVocabCount(words)

print("----- Brown details WITHOUT stopwords -----")
for c in brown.categories():
    words = brown.words(categories=c)

    filtered = []
    for w in words:
        if w not in stopwords:
            filtered.append(w)

    print("category:", c)
    print("word count:", len(filtered))

    PrintTagandVocabCount(filtered)

print("----- Brown details WITHOUT stopwords and lemmatization -----")
for c in brown.categories():
    words = brown.words(categories=c)

    filtered = []
    lemmatizer = WordNetLemmatizer()
    for word in words:
        w = lemmatizer.lemmatize(word)
        if w not in stopwords:
            filtered.append(w)

    print("category:", c)
    print("word count:", len(filtered))

    PrintTagandVocabCount(filtered)

print("----- Brown details WITHOUT stopwords and stemming -----")
for c in brown.categories():
    words = brown.words(categories=c)

    filtered = []
    stemmer = PorterStemmer()
    for word in words:
        w = stemmer.stem(word)
        if w not in stopwords:
            filtered.append(w)

    print("category:", c)
    print("word count:", len(filtered))

    PrintTagandVocabCount(filtered)