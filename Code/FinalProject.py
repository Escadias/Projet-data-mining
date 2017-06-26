#!/usr/bin/python

#import csv
#from nltk.stem import PorterStemmer
#from collections import OrderedDict
import pandas
import re
import numpy
import math
#import tensorflow as tf
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from gensim.models import word2vec
import logging
from StemmingHelper import StemmingHelper
#from sklearn.naive_bayes import GaussianNB 


#porter=PorterStemmer()
                


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



def stemmingReviews(listAllReview):
    for i in range(len(listAllReview)):
        review=str(listAllReview[i]).split()
        review=[StemmingHelper.stem(word.lower()) for word in review]
        listAllReview[i]=" ".join(review)    
    return listAllReview
    
        
    
        
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
listAllReview=deleteStopWords(listAllReview, listStopWords)
listAllReview = stemmingReviews(listAllReview)
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
matrixYLearn=matrixY[:nbLearning]

#from sklearn.naive_bayes import GaussianNB
#model=GaussianNB().fit(matrixXLearn[list(matrixX)], rateLearning)
#predicted=model.predict(matrixXTest)

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
#listPredict=verifyPredict(predicted, rateTesting)
#print(listPredict)
#print(matrixX.shape)


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)




clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(matrixXLearn, rateLearning)                         
MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
predict = clf.predict(matrixXTest)
print("Number of mislabeled points out of a total %d points : %d , efficiency : "
      % (matrixXTest.shape[0],(rateTesting != predict).sum()), ((rateTesting == predict).sum())/matrixXTest.shape[0]*100)

#import gensim
#from gensim.keyedvectors import KeyedVectors
#word_vectors=KeyedVectors.load_word2vec_format('word2vec.txt', binary=False)
#print(word_vectors[0])

#model=gensim.models.KeyedVectors.load_word2vec_format('word2vec.txt', binary=False)





















#alldataFrame=replacePonctuation(alldataFrame)
#alldataFrame=deleteStopWords(alldataFrame, listStopWords)
#dic=fillDictionary(alldataFrame)
#print(dic)

    





#dicLearnX=fillDictionary(dataFrameLearning)
#xDataFrameLearn=pandas.DataFrame(index=range(max(numpy.shape(dataFrameLearning))), columns=dicLearnX.keys())
#xDataFrameLearn=xDataFrameLearn.fillna(0)
#matrixXLearn=fillX(xDataFrameLearn, dataFrameLearning)
#yDataFrameLearn=pandas.DataFrame(index=range(max(numpy.shape(dataFrameLearning))), columns=range(5))
#yDataFrameLearn=yDataFrameLearn.fillna(0)
#matrixYLearn=fillY(yDataFrameLearn, dataFrameLearning)
#
#dicTestX=fillDictionary(dataFrameTesting)
#xDataFrameTest=pandas.DataFrame(index=range(max(numpy.shape(dataFrameTesting))), columns=dicTestX.keys())
#xDataFrameTest=xDataFrameTest.fillna(0)
#matrixXTest=fillX(xDataFrameTest, dataFrameTesting)
#yDataFrameTest=pandas.DataFrame(index=range(max(numpy.shape(dataFrameTesting))), columns=range(5))
#yDataFrameTest=yDataFrameTest.fillna(0)
#matrixYTest=fillY(yDataFrameTest, dataFrameTesting)






#clf=MultinomialNB().fit(numpy.transpose(matrixXLearn), matrixYLearn)
#predictedNaiveBayes=clf.predict(numpy.transpose(matrixXTest))
#print(predictedNaiveBayes)


#FAIRE UNE REGRESSION

#model = GaussianNB()
#model.fit(da)

#nbWords=countWordsReview(dataFrameLearning)
#probScore1=computeProbScore(dataFrameLearning, 1)/nbWords
#probScore2=computeProbScore(dataFrameLearning, 2)/nbWords
#probScore3=computeProbScore(dataFrameLearning, 3)/nbWords
#probScore4=computeProbScore(dataFrameLearning, 4)/nbWords
#probScore5=computeProbScore(dataFrameLearning, 5)/nbWords
#
#dicLearningScore1=fillDictionaryScore(dataFrameLearning, 1)
#dicLearningScore2=fillDictionaryScore(dataFrameLearning, 2)
#dicLearningScore3=fillDictionaryScore(dataFrameLearning, 3)
#dicLearningScore4=fillDictionaryScore(dataFrameLearning, 4)
#dicLearningScore5=fillDictionaryScore(dataFrameLearning, 5)
#
#
#
#nbWordsScore1=countWordsScore(dicLearningScore1)
#nbWordsScore2=countWordsScore(dicLearningScore2)
#nbWordsScore3=countWordsScore(dicLearningScore3)
#nbWordsScore4=countWordsScore(dicLearningScore4)
#nbWordsScore5=countWordsScore(dicLearningScore5)
#
#dicProbScore1=computeProbWords(dicLearningScore1, nbWordsScore1)
#dicProbScore2=computeProbWords(dicLearningScore2, nbWordsScore2)
#dicProbScore3=computeProbWords(dicLearningScore3, nbWordsScore3)
#dicProbScore4=computeProbWords(dicLearningScore4, nbWordsScore4)
#dicProbScore5=computeProbWords(dicLearningScore5, nbWordsScore5)



