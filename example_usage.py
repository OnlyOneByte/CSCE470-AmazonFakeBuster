# Required imports
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
import pickle

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import string

# imports only for this demo
import random


# loads in the classifier
clf = pickle.load(open("pickles/full_dataset_classifier.pkl", "rb"))


# review parameters to test:
rating = 1
product_category = "PC"
verified_purchase = "N"
review_text = "I am extremely unhappy with this fudge bar! ALL FUDGE BARS SHOULD COME WITH TWO KITTENS, AT LEAST! I DEMAND A REFUND!"




print("Raw Review:")
print("Rating:", rating)
print("Category:", product_category)
print("verified purchase:", verified_purchase)
print("review_text:", review_text)
print("==================================================")


# parsing the text
table = str.maketrans({key: None for key in string.punctuation})
def parseReviewText(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.translate(table)
    filtered_tokens=[]
    lemmatized_tokens = []
    for w in text.split(" "):
        if w not in stop_words:
            lemmatized_tokens.append(lemmatizer.lemmatize(w.lower()))
        filtered_tokens = [' '.join(l) for l in nltk.bigrams(lemmatized_tokens)] + lemmatized_tokens
    return filtered_tokens


parsed_review_text = parseReviewText(review_text)


# creating the prediction vector.
vector = {}

# rating
vector["R"] = rating

# product category
if product_category not in vector:
    vector[product_category] = 1
else:
    vector[product_category] = +1

# Verified Purchase
if verified_purchase == "N":
    vector["VP"] = 0
else:
    vector["VP"] = 1

# text
for token in parsed_review_text:
    if token not in vector:
        vector[token] = 1
    else:
        vector[token] = +1

print("Prediction vector:")
print(vector)
print("========================================================")

# actually predicting
prediction = clf.classify(vector)
print("Prediction:", prediction)