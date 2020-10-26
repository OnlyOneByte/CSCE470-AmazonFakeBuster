from main import Dataset, Labels
from utils import evaluate
import os
import math
import sys
from collections import Counter
K = 5

class KNN:
    def __init__(self):
        # bag of words document vectors
        self.bow = []

    def train(self, ds):
        # print("training")
        allUniqueWords = []

        # collects all unique words
        for ind in ds.index:
            text = ds['REVIEW_TEXT'][ind].split(' ') #array
            title = ds['REVIEW_TITLE'][ind].split(' ') #array
            # rating = ds['RATING'][ind] #int
            # label = ds['LABEL'][ind]
            # verified_purchase = ds['VERIFIED_PURCHASE'][ind]
            for word in text:
                if word not in allUniqueWords:
                    allUniqueWords.append(word)
            for word in title:
                if word not in allUniqueWords:
                    allUniqueWords.append(word)

        # constructs the vector bag of words representation
        for reviewID,ind in enumerate(ds.index):  # https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/
            text = ds['REVIEW_TEXT'][ind].split(' ') #array
            title = ds['REVIEW_TITLE'][ind].split(' ') #array
            # rating = ds['RATING'][ind] #int

            label = Labels.label1 if(ds['LABEL'][ind] == "__label1__") else Labels.label2
            
            # verified_purchase = ds['VERIFIED_PURCHASE'][ind]
            self.bow.append((label, {}))
            occurences = Counter(text)

            # TODO add ratings for every document in model
            # TODO add review_title in model
            # TODO add is_verified_purchase in model

            # loops through every word for every document
            for i in range(0, len(allUniqueWords)):
                word = allUniqueWords[i]
                if word in occurences:
                    self.bow[reviewID][1][word] = occurences[word]


    def predict(self, x):
        words = x
        occurences = Counter(words)
        total = []
        den1 = 0

        #gets Ai term
        for k in occurences.keys():
            den1 += occurences[k]**2

        for j in range(len(self.bow)):
            num = 0
            den2 = 0
            #gets Bi term
            for b in self.bow[j][1].keys():
                den2 += self.bow[j][1][b] **2

            #gets numerator
            #for each word in the doument
            for key in occurences.keys():
                #if it also exists in BOW
                if(key in self.bow[j][1]):
                    match = [key,self.bow[j][1][key]]
                    # occurencesBOW * occurencesDocument
                    num += match[1] * occurences[key]

            #gets total for each document
            cosSim = num / (math.sqrt(den1) * math.sqrt(den2))
            total.append([cosSim, self.bow[j][0]])


        #gets k nearest
        total.sort()
        total = total[-K:]

        answer = {}
        for t in total:
            if t[1] in answer:
                answer[t[1]] +=1
            else:
                answer[t[1]] = 1
        
        #gets label
        return max(answer, key=answer.get)




def main(train_split):
    knn = KNN()
    ds = Dataset().fetch()
    knn.train(ds)

    # Evaluate the trained model on training data set.
    evaluate(knn, ds)


    # students should ignore this part.
    # test dataset is not public.
    # only used by the grader.
    if 'GRADING' in os.environ:
        print('\n' + '-'*20 + ' TEST ' + '-'*20)
        test_ds = Dataset('test').fetch()
        evaluate(knn, test_ds)


if __name__ == "__main__":
    train_split = 'train'
    if len(sys.argv) > 1 and sys.argv[1] == 'train_half':
        train_split = 'train_half'
    main(train_split)
