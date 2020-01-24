import nltk
# make sure that everything for nltk is installed.
nltk.download('stopwords')
nltk.download('brown')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# import downloaded
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import brown
from collections import Counter


def PrintDetails(category, words):
    print("Category: ", category)
    print("Token count: ", len(words))

    # get tags via nltk
    tags = nltk.pos_tag(words)

    # list of all unique tags and their count
    ltags = []  # format [["tag", x], ...]

    # loop through all tags
    for t in tags:
        tag = t[1].lower()  # get the tag from object

        # bool for if the tag should be appended to the list of tags
        unique = True

        # loop through current list of tags
        for li in ltags:
            # tag is not unique
            if li[0].lower() == tag:
                unique = False
                li[1] += 1
                break

        if unique:
            ltags.append([tag, 1])

    print("Word-type count: ", len(ltags))
    print("Vocabulary size of category: ", len(set(words)))
    print("")


stopwords = stopwords.words('english')

# Note. I loop through the categories multiple times because I want
# to keep the questions seperated.

print("----- a) Brown details with stopwords -----")
for c in brown.categories():
    words = brown.words(categories=c)
    PrintDetails(c, words)

print("----- b) Brown details WITHOUT stopwords -----")
for c in brown.categories():
    words = brown.words(categories=c)

    filtered = []
    for w in words:
        if w not in stopwords:
            filtered.append(w)

    PrintDetails(c, filtered)

print("----- c) Brown details WITHOUT stopwords and lemmatization -----")
for c in brown.categories():
    words = brown.words(categories=c)

    filtered = []
    lemmatizer = WordNetLemmatizer()
    for word in words:
        w = lemmatizer.lemmatize(word)
        if w.lower() not in stopwords:
            filtered.append(w)

    PrintDetails(c, filtered)

print("----- d) Brown details WITHOUT stopwords and stemming -----")
for c in brown.categories():
    words = brown.words(categories=c)

    filtered = []
    stemmer = PorterStemmer()
    for word in words:
        w = stemmer.stem(word)
        if w.lower() not in stopwords:
            filtered.append(w)

    PrintDetails(c, filtered)

print("Vocabulary size of the whole corpus:", len(set(brown.words())))
