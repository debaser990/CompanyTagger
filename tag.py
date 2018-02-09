import pandas
#import matplotlib.pyplot as plt
#from sklearn import model_selection
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.metrics import classification_report
#import numpy
#from sklearn.feature_extraction.text import CountVectorizer
import nltk
#from nltk.tokenize.moses import MosesTokenizer, MosesDetokenizer 
import string
from nltk import NaiveBayesClassifier


#url = "C:\Users\Akshat\Documents\dumps\similar_companies_data.csv"
url = "C:\Users\Akshat\Documents\dumps\synonyms11.csv"
names = ['feat', 'lab']
dataset = pandas.read_csv(url, names=names)
print(dataset.shape) 

array = dataset.values
X = array[:,0:1]
Y = array[:,1:2]


# for cross-validation
#validation_size = 0.20
#seed = 7
#X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#seed = 7
#X_feat = X_train.tolist()
#Y_feat = Y_train.tolist()

X_feat = X.tolist()
Y_feat = Y.tolist()

X_flat=[]
Y_flat=[]
xl=[]
yl=[]

for x in X_feat:
    for y in x:
        X_flat.append(y)    #flatten
for x1 in Y_feat:
    for y1 in x1:
        Y_flat.append(y1)
for x2 in X_flat:
    xl.append(x2.lower())    #low
for x3 in Y_flat:
    yl.append(x3.lower())
    
#print len(xl)
#print len(yl)

def feat_gen(x):
    
    dic = {}
    bag = ['pvt','private','ltd','limited','Bpo','Pvt','Private','Limited','india','India']
    i = -1
    for entry in x:
        try:
            #print type(x)
            #print x
            #break
            #m,d = MosesTokenizer(), MosesDetokenizer()
            #token=m.tokenize(x)
            token = nltk.word_tokenize(x) # x is a list to iterate thru and cut
            
            #print token
            
        except TypeError as te:
            print entry
            print type(entry)
            print te
            

        for el in bag:

            if el in token:
                i = token.index(el)
                token=token[0:i]
                t="".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in token]).strip()
                #print t
                #print t
                #print token
                dic={t:True}
                #print dic
                #break
            else:
                pass
        

    return dic

#DatList= array.tolist()

DatList=zip(xl,yl)
featuresets=[(feat_gen(x1),y1) for (x1,y1) in DatList]

#print featuresets

train_set, test_set = featuresets[500:], featuresets[:500]


#print type(featuresets)
classifier = NaiveBayesClassifier.train(train_set)
#print nltk.classify.accuracy(classifier, test_set)
ClassifiedName=classifier.classify(feat_gen('Infosys private limited'))
print ClassifiedName
