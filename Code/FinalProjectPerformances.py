#!/usr/bin/python

import pandas
import re
import numpy
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from gensim.models import word2vec
import logging
from sklearn import svm
from StemmingHelper import StemmingHelper

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
    
def baseLine(dataframe):
    res = numpy.zeros((1,5))
    print('Based on ', len(dataframe), 'reviews')
    for i in range(1,6):
        res[0][i-1] = len(dataframe[dataframe['user_rating']==i])
        print('number of ', i,' stars : ', res[0][i-1], ' ', res[0][i-1]/len(dataframe)*100 ,'%')
    return res


def fillX(dataFrame, listAllReview):
    for i in range (len(listAllReview)):
        for word in listAllReview[i].split():
            dataFrame[word][i]+=1
    return dataFrame

def fillY(dataFrame, listAllRating):
    for i in range(len(listAllRating)):
        dataFrame[listAllRating[i]-1][i]=1
    return dataFrame 

def verifyPredict(predicted, rateTesting):
    goodClassification=0
    badClassification=0
    nbClassification=0
    for i in range (len(predicted)):
        nbClassification+=1
        if(predicted[i]==rateTesting[i]):
            goodClassification+=1
        else:
            badClassification+=1
    return [goodClassification/nbClassification, badClassification/nbClassification]

def averageReview(dicZ, matrixX):
    matAvg=[]
    for i in range(matrixX.shape[0]):
        avg=[0]*200
        countWords=0
        for column in list(matrixX):
            if(column in dicZ.keys()):
                if(matrixX[column][i]!=0):
                    avg=avg+(numpy.dot(dicZ.get(column),matrixX[column][i]))
                    countWords+=matrixX[column][i]
        if(countWords!=0):            
            avg=numpy.dot(avg,1/countWords)
        matAvg.append(avg)
    return matAvg

def fillZ(dic, wordVectors, listWords):
    for i in range(len(listWords)):
        if(listWords[i] in wordVectors.vocab.keys()):
            dic[listWords[i]]=wordVectors[listWords[i]]
    return dic

def normalizeMatrix(mat):
    matNorm=numpy.reshape(mat,(mat.shape[0], mat.shape[1]))
    
    minValue=999
    maxValue=-999
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if(mat[j][i]<minValue):
                minValue=mat[j][i]
            if(mat[j][i]>maxValue):
                maxValue=mat[j][i]
    
    for i in range (mat.shape[0]):
        for j in range (mat.shape[1]):
            matNorm[j][i]=(mat[j][i]-minValue)/(maxValue-minValue)
    return matNorm

listMNBSZ=[]
listMNBAZ=[]
listNNSZ=[]
listNNAZ=[]
listSVMSZ=[]
listSVMAZ=[]
listTimeMNBSZ=[]
listTimeMNBAZ=[]
listTimeNNSZ=[]
listTimeNNAZ=[]
listTimeSVMSZ=[]
listTimeSVMAZ=[]


for nbEssai in range(3):       
    print(nbEssai)
    listStopWords=[]
    
    #Ouverture des fichiers
    fileStopWords=open('stop_words.txt','r')
    fileAlwayscsv=pandas.read_csv('datasets_clean/reviews_always.csv')#,sep='\t', encoding='latin-1')
    fileGillettecsv=pandas.read_csv('datasets_clean/reviews_gillette.csv')#,sep='\t', encoding='latin-1')
    fileOralbcsv=pandas.read_csv('datasets_clean/reviews_oral-b.csv')#,sep='\t', encoding='latin-1')
    filePantenecsv=pandas.read_csv('datasets_clean/reviews_pantene.csv')#,sep='\t', encoding='latin-1')
    fileTampaxcsv=pandas.read_csv('datasets_clean/reviews_tampax.csv')#,sep='\t', encoding='latin-1')
    
    
    #frames = [fileAlwayscsv, fileOralbcsv, filePantenecsv, fileTampaxcsv]
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
    
    
    
    matrixX=fillX(xDataFrame, listAllReview)
    
    matrixY=fillY(yDataFrame, listAllRating)
    
    
    matrixXLearn=matrixX[:nbLearning]
    matrixXTest=matrixX[nbLearning:]
    
    from gensim.models.keyedvectors import KeyedVectors
    wordVectors=KeyedVectors.load_word2vec_format('word2vec.txt', binary=False)
    
    dicZ={}
    dicZ=fillZ(dicZ, wordVectors, list(dic.keys()))
    matAvg=averageReview(dicZ, matrixX)
    matAvg=pandas.DataFrame(matAvg, columns=list(range(200)))
    matAvgLearn=matAvg[:nbLearning]
    matAvgTest=matAvg[nbLearning:]
    
#    baseLine(alldataFrame)
    
    #############################NAIVE BAYES###########################
    
    debMNBSZ=time.time()
    model=MultinomialNB().fit(matrixXLearn[list(matrixX)], rateLearning)
    finMNBSZ=time.time()
    predictedMNBSZ=model.predict(matrixXTest)
    
    listPredictMNBSZ=verifyPredict(predictedMNBSZ, rateTesting)
    listMNBSZ.append(listPredictMNBSZ)
    
    matAvgNorm=normalizeMatrix(matAvg)
    matAvgNormLearn=matAvgNorm[:nbLearning]
    matAvgNormTest=matAvgNorm[nbLearning:]
    
    debMNBAZ=time.time()
    model=MultinomialNB().fit(matAvgNormLearn[list(matAvg)], rateLearning)
    finMNBAZ=time.time()
    predictedMNBAZ=model.predict(matAvgNormTest)
    
    
    listTimeMNBSZ.append(finMNBSZ-debMNBSZ)
    listTimeMNBAZ.append(finMNBAZ-debMNBAZ)
    
    listPredictMNBAZ=verifyPredict(predictedMNBAZ, rateTesting)
    listMNBAZ.append(listPredictMNBAZ)
    
    ###################################################################
    
    
    
    ############################Neural Network########################
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    clf = MLPClassifier(solver='sgd', alpha=1e-5,
                         hidden_layer_sizes=(5, 2), random_state=1)
    
    debutNNSZ=time.time()
    clf.fit(matrixXLearn, rateLearning)                         
    
    finNNSZ=time.time()
    predict = clf.predict(matrixXTest)
    
    
    listNNSZ.append(((rateTesting == predict).sum())/matrixXTest.shape[0]*100)
    
    debutNNAZ=time.time()
    clf.fit(matAvgLearn, rateLearning)                         
    
    finNNAZ=time.time()
    
    listTimeNNSZ.append(finNNSZ-debutNNSZ)
    listTimeNNAZ.append(finNNAZ-debutNNAZ)
    
    predictZ = clf.predict(matAvgTest)
    listNNAZ.append(((rateTesting == predictZ).sum())/matAvgTest.shape[0]*100)
    
    ###################################################################
    
    
    ########################### SVM ###########################
    
    
    clf=svm.SVC()
    debutSVMSZ=time.time()
    clf.fit(matrixXLearn[list(matrixX)], rateLearning)
    finSVMSZ=time.time()
    
    predictedSVM=clf.predict(matrixXTest)
    listPredictSVM=verifyPredict(predictedSVM, rateTesting)
    listSVMSZ.append(listPredictSVM)
    
    clf2=svm.SVC()
    debutSVMAZ=time.time()
    clf2.fit(matAvgLearn[list(matAvg)], rateLearning)
    finSVMAZ=time.time()
    predictedSVM2=clf2.predict(matAvgTest)
    listPredictSVM2=verifyPredict(predictedSVM2, rateTesting)
    listSVMAZ.append(listPredictSVM2)
    
    listTimeSVMSZ.append(finSVMSZ-debutSVMSZ)
    listTimeSVMAZ.append(finSVMAZ-debutSVMAZ)

print(listMNBSZ, listMNBAZ, listNNSZ, listNNAZ, listSVMSZ, listSVMAZ, listTimeMNBSZ, listTimeMNBAZ, listTimeNNSZ, listTimeNNAZ, listTimeSVMSZ, listTimeSVMAZ)
