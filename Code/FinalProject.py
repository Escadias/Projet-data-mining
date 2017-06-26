#!/usr/bin/python

#import csv
from nltk.stem import PorterStemmer
#from collections import OrderedDict
import pandas
import re
import numpy



#from sklearn.naive_bayes import GaussianNB 

porter=PorterStemmer()
                


def replacePonctuation(listAllReview):
    for i in range(len(listAllReview)):
        listAllReview[i]=str(re.sub("[^a-zA-Z]"," ",str(listAllReview[i])))
    return listAllReview
   
def fillDictionary(listAllReview):
    dic={}
    for review in listAllReview:
        words=review.split()
        for word in words:
            if(dic.get(word)!=None):
                dic[word]=dic.get(word)+1
            else:
                dic[word]=1
    return dic



def deleteStopWords(listAllReview, listStopWords):
    for i in range(len(listAllReview)):
        review=str(listAllReview[i]).split()
        review=[word.lower() for word in review if word.lower() not in listStopWords]
        listAllReview[i]=" ".join(review)
    return listAllReview



def modifyWordPorter(listAllReview):
    listReview=listAllReview
    for j in range (len(listReview)):
        review=listReview[j].split()
        for i in range(len(review)):
            review[i]=porter.stem(review[i])
        listReview[j]=" ".join(review)
    return listReview

        
    
        
listStopWords=[]

#Ouverture des fichiers
fileStopWords=open('stop_words.txt','r')
fileAlways=open('datasets_clean/reviews_always.csv','r')
fileAlwayscsv=pandas.read_csv(fileAlways)
fileGillette=open('datasets_clean/reviews_gillette.csv','r')
fileGillettecsv=pandas.read_csv(fileGillette)
fileOralb=open('datasets_clean/reviews_oral-b.csv','r')
fileOralbcsv=pandas.read_csv(fileOralb)
filePantene=open('datasets_clean/reviews_pantene.csv','r')
filePantenecsv=pandas.read_csv(filePantene)
fileTampax=open('datasets_clean/reviews_tampax.csv','r')
fileTampaxcsv=pandas.read_csv(fileTampax)

frames = [fileAlwayscsv, fileGillettecsv, fileOralbcsv, filePantenecsv, fileTampaxcsv]
alldataFrame = pandas.concat(frames)
alldataFrame = alldataFrame.sample(frac= 1)
listAllReview=list(alldataFrame['review'].astype(str))
listAllRating=list(alldataFrame['user_rating'].astype(int))

for row in fileStopWords:
    listStopWords.append(str(row.rstrip('\n')).lower())



listAllReview=replacePonctuation(listAllReview)
#listAllReview=modifyWordPorter(listAllReview)
listAllReview=deleteStopWords(listAllReview, listStopWords)
nbLearning = round(len(listAllReview)*2/3)

reviewLearning=listAllReview[:nbLearning]
rateLearning=listAllRating[:nbLearning]
reviewTesting=listAllReview[nbLearning:]
rateTesting=listAllRating[nbLearning:]

dic=fillDictionary(listAllReview)
dicLearn=fillDictionary(reviewLearning)
dicTest=fillDictionary(reviewTesting)

xDataFrame=pandas.DataFrame(index=range(len(listAllReview)), columns=dic.keys())
xDataFrame=xDataFrame.fillna(0)

yDataFrame=pandas.DataFrame(index=range(len(listAllReview)), columns=range(5))
yDataFrame=yDataFrame.fillna(0)

def fillX(dataFrame, listAllReview):
    for i in range (len(listAllReview)):
        for word in listAllReview[i].split():
            dataFrame[word][i]+=1
    return dataFrame


matrixX=fillX(xDataFrame, listAllReview)

def fillY(dataFrame, listAllRating):
    for i in range(len(listAllRating)):
        dataFrame[listAllRating[i]-1][i]=1
    return dataFrame 

matrixY=fillY(yDataFrame, listAllRating)


matrixXLearn=matrixX[:nbLearning]
matrixXTest=matrixX[nbLearning:]

#from sklearn.naive_bayes import GaussianNB
#model=GaussianNB().fit(matrixXLearn[list(matrixX)], rateLearning)
#predicted=model.predict(matrixXTest)

#from sklearn.naive_bayes import MultinomialNB
#model=MultinomialNB().fit(matrixXLearn[list(matrixX)], rateLearning)
#predicted=model.predict(matrixXTest)
#
#
#
#
#def verifyPredict(predicted, rateTesting):
#    goodClassification=0
#    badClassification=0
#    nbClassification=0
#    for i in range (len(predicted)):
#        nbClassification+=1
#        if(predicted[i]==rateTesting[i]):
#            goodClassification+=1
#        else:
#            badClassification+=1
#    return [goodClassification/nbClassification, badClassification/nbClassification]
#
#
#listPredict=verifyPredict(predicted, rateTesting)
#print(listPredict)


from gensim.models.keyedvectors import KeyedVectors
wordVectors=KeyedVectors.load_word2vec_format('word2vec.txt', binary=False)
print('lkshjflkguhekurhhf' in wordVectors.vocab.keys())

def fillZ(dic, wordVectors, listWords):
    for i in range(len(listWords)):
        if(listWords[i] in wordVectors.vocab.keys()):
            dic[listWords[i]]=wordVectors[listWords[i]]
    return dic


    













