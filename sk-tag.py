import pandas
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import string
from nltk import NaiveBayesClassifier
from sets import Set
import numpy as np
from nltk.classify.scikitlearn import SklearnClassifier

#url = "C:\Users\Akshat\Documents\dumps\similar_companies_data.csv"
url = "C:\Users\Akshat\Documents\dumps\synonyms11.csv"
names = ['feat', 'lab']
dataset = pandas.read_csv(url, names=names)
print(dataset.shape) 

array = dataset.values
X = array[:,0:1]
Y = array[:,1:2]


# for cross-validation
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
seed = 7

X_feat = X_train.tolist()
Y_feat = Y_train.tolist()

X_test = X_validation.tolist()
Y_test = Y_validation.tolist()
#X_feat = X.tolist()
#Y_feat = Y.tolist()

X_flat=[]
Y_flat=[]
xl=[]
yl=[]

#training set
for x in X_feat:
    for y in x:
        X_flat.append(y)    #flattening train
for x1 in Y_feat:
    for y1 in x1:
        Y_flat.append(y1)
for x2 in X_flat:
    xl.append(x2.lower())    #lowered xl->unfiltered
for x3 in Y_flat:
    yl.append(x3.lower())
    
#print X_test
#print Y_test

X_flat=[]
Y_flat=[]
xs=[]
ys=[]

#validation set
for x1 in X_test:
    for y1 in x1:
        X_flat.append(y1)
for x2 in Y_test:
    for y2 in x2:
        Y_flat.append(y2)
for x2 in X_flat:
    xs.append(x2.lower())    #lowered xs
for x3 in Y_flat:
    ys.append(x3.lower())        
        
tokenList = []

def feat(x):

    bag = Set(['pvt','private','ltd','limited','Bpo','Pvt','Private','Limited'])
    
    index = -1
    for i in x:
        tokenList.append(nltk.word_tokenize(i))
        #print tokenList
            
    for i in range(len(tokenList)):
        li = tokenList[i]
        for word in li:

            if word in bag :
                index = li.index(word)
                break

        tokenList[i] = li[0:index]

    return tokenList

xt = feat(xl)
#print xt

def detokenize(y):
    for u in y:
        uc = "".join([" " +r if not r.startswith("'") and r not in string.punctuation else r for r in u]).strip()
        lis.append(uc)
    return lis

xtrain = detokenize(xt)

vectorizer = CountVectorizer(decode_error = 'ignore', lowercase=False)
vectorizer.fit(xtrain)
vecTrain = vectorizer.transform(xtrain)
vecTest = vectorizer.transform(xs)
clf = MultinomialNB()
clf.fit(vecTrain,yl)
predicted = clf.predict(vecTest)
print predicted
print "AND THE LABELS : \n"
print ys

