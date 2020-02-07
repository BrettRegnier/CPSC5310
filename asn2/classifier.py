from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import brown
import random
import nltk
nltk.download("brown")  # just incase


# TODO
# Take each word from each category and classify them as their category.
# train the naive bayes
# Feed in a whole document and let it classify what category

# OR Tokenize each word in a text, from the brown corpus, and take its category and create training data from that
# similar to how the examples in class are. From there we can feed in texts that need to be classified. Should work better.

def main():
    features = None
    shuffles = 0
    
    choice = ""
    while choice != "exit":
        choice = ""
        while choice not in ["run", "exit"]:
            choice = input("Would you like to classify the Brown corpus? [Run, Exit]\n>").lower()
        
        if choice == "exit":
            break

        shuffles = 0
        while shuffles < 1 or shuffles > 100:
            shuffles = int(input("How many times would you like to shuffle and run? [1-100]\n>"))
            print("")
    
        # Create features if not already made
        if features is None:
            print("Creating features...")
            features = CreateFeatures()

        # shuffle, train and test up to x amount
        for i in range(shuffles):
            print("Iteration:", i)
            random.shuffle(features)

            # split the data into training and data
            split = int(len(features) * .7)
            trainingData = features[:split]
            testingData = features[split:]

            # train the classifier
            print("  Training Classifier...")
            classifier = nltk.NaiveBayesClassifier.train(trainingData)
            print("  DING! - Done training!")
            print("  Training set:", len(trainingData), "items")
            print("  Testing set:", len(testingData), "items")
            print("  Classifying testing data...")
            accuracy = nltk.classify.accuracy(classifier, testingData) * 100 # turn into percent.
            print("  Accuracy:", "{0:.4f}".format(round(accuracy,4)) + "%")
            print("")


def CreateFeatures():
    features = []
    sw = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()  # yielded an extra 0.1% so I kept it in.
    
    for c in brown.categories():
        for d in brown.fileids(categories=c):
            words = brown.words(fileids=d)
            extracted_words = ""
            for w in words:
                w = lemmatizer.lemmatize(w)
                w = stemmer.stem(w)
                if (w not in sw) and (w.isalnum()):
                    extracted_words += w.lower() + " "
                    # feature_words.append({"w": w, "c": c})
                    # tokens.append(w)
                feature = ({"words": extracted_words}, c)
                features.append(feature)

    return features


if __name__ == "__main__":
    main()
