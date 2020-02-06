import nltk
import random
from nltk.corpus import brown
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

# TODO
# Take each word from each category and classify them as their category.
# train the naive bayes
# Feed in a whole document and let it classify what category

# OR Tokenize each word in a text, from the brown corpus, and take its category and create training data from that
# similar to how the examples in class are. From there we can feed in texts that need to be classified. Should work better.

# create features based on the sentence and the category it is associated with
stopwords = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
features = []
for c in brown.categories():
    sentences = list(brown.sents(categories=c))
    sentence = ""
    for s in sentences:
        # turn the tokenized sentence into a alpha numeric without punctuation
        # and remove stopwords
        for word in s:
            word = lemmatizer.lemmatize(word)
            if (word not in stopwords) and (word.isalnum()):
                sentence += word.lower() + " "
                feature = ({"word": word}, c)
                features.append(feature)
        # sentence = ' '.join([e for e in s if e.isalpha()])
        # feature = ({"word": word}, c)
        # features.append(feature)

# shuffle the data so the classifier will recieve a different set of
# training and testing each time.
random.shuffle(features)

# split into 70% and 30%
trainingSize = int(len(features) * .7)
trainingData = features[:trainingSize]
testingData = features[trainingSize:]

# train the classifier
classifier = nltk.NaiveBayesClassifier.train(trainingData)
print("Done training")

print(classifier.classify(
    {'sentence': 'dan morgan told would forget ann turner'}))

print("Accuracy on testing data:", nltk.classify.accuracy(classifier, testingData))
