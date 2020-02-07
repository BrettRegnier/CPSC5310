from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import brown
import random
import nltk
nltk.download("brown")  # just incase


def main():
    features = None
    shuffles = 0

    choice = ""
    while choice != "exit":
        choice = ""
        while choice not in ["run", "exit"]:
            choice = input(
                "Would you like to classify the Brown corpus? [Run, Exit]\n>").lower()

        if choice == "exit":
            break

        shuffles = 0
        while shuffles < 1 or shuffles > 100:
            shuffles = int(
                input("How many times would you like to shuffle and run? [1-100]\n>"))
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
            # turn into percent.
            accuracy = nltk.classify.accuracy(classifier, testingData) * 100
            print("  Accuracy:", "{0:.4f}".format(round(accuracy, 4)) + "%")
            print("")


def CreateFeatures():
    features = []
    sw = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()  # yielded an extra 0.1% so I kept it in.

    # get the categories in brown
    for c in brown.categories():
        # get the files in a category
        for d in brown.fileids(categories=c):
            # get words from a file
            words = brown.words(fileids=d)

            # extracted words of a document appended together.
            extracted_words = ""

            # filter each word
            for w in words:
                w = lemmatizer.lemmatize(w)
                w = stemmer.stem(w)
                if (w not in sw) and (w.isalnum()):
                    # append until we have the filtered document recreated
                    extracted_words += w.lower() + " "

                # create features and add them to a feature set.
                feature = ({"words": extracted_words}, c)
                features.append(feature)

    return features


if __name__ == "__main__":
    main()
