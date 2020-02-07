import random

import nltk
from nltk import probability

from nltk.corpus import brown

#remove punctuation

def main():
    shannonWords = None
    shannonSentences = None
    bigrams = None
    trigrams = None
    
    cfd_bigram = None
    cfd_trigram = None
    
    choice = ""
    while choice != "exit":
        choice = ""
        while choice not in ["bigram", "trigram", "exit"]:
            choice = input("Which kind of sentence would you like to generate? [Bigram, Trigram, Exit]\n>").lower()
        
        if choice == "exit":
            break

        gen = 0
        while gen < 1 or gen > 100:
            inp = input("How many times would you like to shuffle and run? [1-100]\n>")
            if inp.isnumeric():
                gen = int(inp)
            print("")


        if shannonSentences is None and shannonWords is None:
            print("Generating shannon sentences...")
            shannonSentences = []
            shannonWords = []
            
            sentences = brown.sents()
            length = len(sentences)
            
            # clean out the punctuation
            for s in sentences:
                # create a sentence list built as a shannon sentence
                sentence = []
                sentence.append("<s>")
                shannonWords.append("<s>")
                for w in s:
                    # only append words and numbers but not punctuation
                    if w.isalnum():
                        sentence.append(w)
                        shannonWords.append(w)
                sentence.append("</s>")
                shannonWords.append("</s>")
                
                # so a random sentence can be picked for trigrams
                shannonSentences.append(sentence)

        if choice == "bigram":
            if cfd_bigram is None:
                print("Generating bigram conditional frequency distributions...")
                bigrams = nltk.bigrams(shannonWords)
                cfd_bigram = nltk.ConditionalFreqDist(bigrams)
        elif choice == "trigram":
            if cfd_trigram is None:
                print("Generating trigram conditional frequency distributions...")
                trigrams = nltk.ngrams(shannonWords, 3)
                conditionals = []
                for w0, w1, w2 in trigrams:
                    conditionals.append((w0 + " " + w1, w2))
                trigrams = conditionals

                cfd_trigram = nltk.ConditionalFreqDist(trigrams)

        for i in range(gen):
            s = ""
            if choice == "bigram":
                s = GenerateBigramSentence(cfd_bigram)
            elif choice == "trigram":
                # randomly pick a starting condition gram from the shannon sentences
                pick = random.randint(0, len(shannonSentences))
                cond = shannonSentences[pick][0] + " " + shannonSentences[pick][1]

                s = GenerateTrigramSentence(cfd_trigram, cond)
            print(str(i+1) + ":", s, end="\n\n")

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
    # start the sentence as the starting gram plus a space.
    sentence = gram + " "
    word = ""
    # while the selected next work is not the end of a sentence
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