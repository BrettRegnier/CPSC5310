import random

import nltk
from nltk import probability

from nltk.corpus import brown

def main():
    choice = ""
    # could be better of UI but its here!
    while choice != "exit":
        choice = ""
        while choice not in ["bigram", "trigram", "exit"]:
            choice = input("Which kind of sentence would you like to generate? [Bigram, Trigram, Exit]\n>").lower()
        
        if choice == "exit":
            break

        gen = 0
        while gen < 1 or gen > 100:
            gen = int(input("How many sentences to generate? [1-100]\n>"))
            print("")


        shannonWords = []
        shannonSentences = []
        sentences = brown.sents()
        length = len(sentences)
        for s in sentences:
            # insert the beginning and ending tags
            s.insert(0, "<s>")
            s.append("</s>")

            # so a random sentence can be picked for trigrams
            shannonSentences.append(s) 

            # append words to the shannonWord list
            for w in s:
                shannonWords.append(w)

        cfd = None
        if choice == "bigram":
            bigrams = nltk.bigrams(shannonWords)
            cfd = nltk.ConditionalFreqDist(bigrams)
        elif choice == "trigram":
            trigrams = nltk.ngrams(shannonWords, 3)
            conditionals = []
            for w0, w1, w2 in trigrams:
                conditionals.append((w0 + " " + w1, w2))

            cfd = nltk.ConditionalFreqDist(conditionals)

        for i in range(gen):
            s = ""
            if choice == "bigram":
                s = GenerateBigramSentence(cfd)
            elif choice == "trigram":
                # randomly pick a starting condition gram from the shannon sentences
                pick = random.randint(0, len(shannonSentences))
                cond = shannonSentences[pick][0] + " " + shannonSentences[pick][1]

                s = GenerateTrigramSentence(cfd, cond)
            print(str(i+1) + ":", s, end="\n\n")

def GetIntChoice(low, high):
    choice = 0
    while choice < low and choice > high:
        choice = raw_input("Ch")

def GenerateBigramSentence(cfd):
    # can start every sentence this way since it will never change for bigram
    sentence = "<s> "
    word = "<s>"
    while word != "</s>":
        rand = random.random()
        total = 0
        for w in cfd[word]:
            total += cfd[word].freq(w)
            if rand <= total:
                word = w
                sentence += word + " " 
                break
    
    return sentence

def GenerateTrigramSentence(cfd, gram):
    sentence = "<s> The "
    gram = "<s> The"
    word = ""
    while word != "</s>":
        rand = random.random()
        total = 0
        for w in cfd[gram]:
            total += cfd[gram].freq(w)
            if rand <= total:
                word = w
                string = gram.split(" ")
                gram = string[1] + " " + word
                sentence += word + " "
                break
    
    return sentence

if __name__ == "__main__":
    main()