#!/usr/bin/python

import pandas
import re
import numpy
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from gensim.models import word2vec
import logging
from StemmingHelper import StemmingHelper

#Supprime la ponctuation sur toutes les reviews
def replacePonctuation(listAllReview):
    for i in range(len(listAllReview)):
        listAllReview[i]=str(re.sub("[^a-zA-Z]"," ",str(listAllReview[i])))
    return listAllReview
   
#Fonction qui crée le dictionnaire (mot, nombre d'occurences de ce mot dans la liste de reviews)
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


#Fonction qui supprime tous les stop words des reviews
def deleteStopWords(listAllReview, listStopWords):
    for i in range(len(listAllReview)):
        review=str(listAllReview[i]).split()
        review=[word.lower() for word in review if word.lower() not in listStopWords]
        listAllReview[i]=" ".join(review)    
    return listAllReview

#Application du stemming sur les reviews
def stemmingReviews(listAllReview):
    for i in range(len(listAllReview)):
        review=str(listAllReview[i]).split()
        review=[StemmingHelper.stem(word.lower()) for word in review]
        listAllReview[i]=" ".join(review)    
    return listAllReview
    
#Compteur des notes (1, 2, 3, 4, 5) 
def baseLine(dataframe):
    res = numpy.zeros((1,5))
    print('Based on ', len(dataframe), 'reviews')
    for i in range(1,6):
        res[0][i-1] = len(dataframe[dataframe['user_rating']==i])
        print('number of ', i,' stars : ', res[0][i-1], ' ', res[0][i-1]/len(dataframe)*100 ,'%')
    return res

#Remplit la matrice X
def fillX(dataFrame, listAllReview):
    for i in range (len(listAllReview)):
        for word in listAllReview[i].split():
            dataFrame[word][i]+=1
    return dataFrame

#Remplit la matrice Y
def fillY(dataFrame, listAllRating):
    for i in range(len(listAllRating)):
        dataFrame[listAllRating[i]-1][i]=1
    return dataFrame 

#Verifie les predictions en se servant d'un ensemble de test
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

#Calcul la moyenne de chaque review en utilisant les vecteurs de word2vec et les mots de chaque review
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

#Remplit la matrice Z (dictionnaire (mot, sémantique sous forme de vecteur))
def fillZ(dic, wordVectors, listWords):
    for i in range(len(listWords)):
        if(listWords[i] in wordVectors.vocab.keys()):
            dic[listWords[i]]=wordVectors[listWords[i]]
    return dic

#Normalise la matrice pour ne plus avoir de valeurs négatives
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

       

listStopWords=[]

#Ouverture des fichiers
fileStopWords=open('stop_words.txt','r')
fileAlwayscsv=pandas.read_csv('datasets_clean/reviews_always.csv')#,sep='\t', encoding='latin-1')
fileGillettecsv=pandas.read_csv('datasets_clean/reviews_gillette.csv')#,sep='\t', encoding='latin-1')
fileOralbcsv=pandas.read_csv('datasets_clean/reviews_oral-b.csv')#,sep='\t', encoding='latin-1')
filePantenecsv=pandas.read_csv('datasets_clean/reviews_pantene.csv')#,sep='\t', encoding='latin-1')
fileTampaxcsv=pandas.read_csv('datasets_clean/reviews_tampax.csv')#,sep='\t', encoding='latin-1')


frames = [fileAlwayscsv, fileGillettecsv, fileOralbcsv, filePantenecsv, fileTampaxcsv]
alldataFrame = pandas.concat(frames)
alldataFrame = alldataFrame.sample(frac= 1)
#Creation des listes de notes et de reviews
listAllReview=list(alldataFrame['review'].astype(str))
listAllRating=list(alldataFrame['user_rating'].astype(int))

#Cree la liste pour les stop words
for row in fileStopWords:
    listStopWords.append(str(row.rstrip('\n')).lower())


#Preprocessing des reviews
listAllReview=replacePonctuation(listAllReview)
listAllReview=deleteStopWords(listAllReview, listStopWords)
listAllReview = stemmingReviews(listAllReview)
nbLearning = round(len(listAllReview)*2/3)

rateLearning=listAllRating[:nbLearning]
rateTesting=listAllRating[nbLearning:]

dic=fillDictionary(listAllReview)
#Creation de la matrice X
xDataFrame=pandas.DataFrame(index=range(len(listAllReview)), columns=dic.keys())
xDataFrame=xDataFrame.fillna(0)
matrixX=fillX(xDataFrame, listAllReview)


#Separation de la matrice X en matrice d'apprentissage et de test
matrixXLearn=matrixX[:nbLearning]
matrixXTest=matrixX[nbLearning:]

from gensim.models.keyedvectors import KeyedVectors
wordVectors=KeyedVectors.load_word2vec_format('word2vec.txt', binary=False)
#Creation de la matrice (dictionnaire) Z et création de la matrice contenant la moyenne des vecteurs word2vec en utilisant les mots de chaque review
dicZ={}
dicZ=fillZ(dicZ, wordVectors, list(dic.keys()))
matAvg=averageReview(dicZ, matrixX)
matAvg=pandas.DataFrame(matAvg, columns=list(range(200)))
#Séparation matrice en matrice d'apprentissage et de test 
matAvgLearn=matAvg[:nbLearning]
matAvgTest=matAvg[nbLearning:]

#baseLine(alldataFrame)

#############################NAIVE BAYES MULTINOMIAL###########################

#Sans Z
model=MultinomialNB().fit(matrixXLearn[list(matrixX)], rateLearning)
predicted=model.predict(matrixXTest)

listPredictMNBSZ=verifyPredict(predicted, rateTesting)
print("Multinomial Naive Bayes:", listPredictMNBSZ[0], " de réussite, ", listPredictMNBSZ[1], " d'échecs.")

#Avec Z
matAvgNorm=normalizeMatrix(matAvg)
matAvgNormLearn=matAvgNorm[:nbLearning]
matAvgNormTest=matAvgNorm[nbLearning:]

model=MultinomialNB().fit(matAvgNormLearn[list(matAvg)], rateLearning)
predicted=model.predict(matAvgNormTest)
listPredictMNBAZ=verifyPredict(predicted, rateTesting)
print("Multinomial Naive Bayes (Avec Z):", listPredictMNBAZ[0], " de réussite, ", listPredictMNBAZ[1], " d'échecs.")

###################################################################



############################Neural Network########################

#Sans Z
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
clf = MLPClassifier(solver='sgd', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(matrixXLearn, rateLearning)                         

predict = clf.predict(matrixXTest)
predictNNSZ=((rateTesting == predict).sum())/matrixXTest.shape[0]*100

print("Number of mislabeled points out of a total %d points : %d , efficiency : "
      % (matrixXTest.shape[0],(rateTesting != predict).sum()), ((rateTesting == predict).sum())/matrixXTest.shape[0]*100)

#Avec Z
clf.fit(matAvgLearn, rateLearning)                         

predictZ = clf.predict(matAvgTest)
print("Number of mislabeled points out of a total %d points : %d , efficiency : "
      % (matAvgTest.shape[0],(rateTesting != predictZ).sum()), ((rateTesting == predictZ).sum())/matAvgTest.shape[0]*100)

predictNNSZ=((rateTesting == predictZ).sum())/matAvgTest.shape[0]*100

###################################################################


########################### SVM ###########################
#Sans Z
from sklearn import svm
clf=svm.SVC()
clf.fit(matrixXLearn[list(matrixX)], rateLearning)
predictedSVM=clf.predict(matrixXTest)
listPredictSVM=verifyPredict(predictedSVM, rateTesting)
print("SVM (Sans Z):", listPredictSVM[0], " de réussite, ", listPredictSVM[1], " d'échecs.")

#Avec Z
clf2=svm.SVC()
clf2.fit(matAvgLearn[list(matAvg)], rateLearning)
predictedSVM2=clf2.predict(matAvgTest)
listPredictSVM2=verifyPredict(predictedSVM2, rateTesting)
print("SVM (Avec Z):", listPredictSVM2[0], " de réussite, ", listPredictSVM2[1], " d'échecs.")

