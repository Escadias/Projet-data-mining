#!/usr/bin/python

import csv
from nltk.stem import PorterStemmer
#from collections import OrderedDict
import pandas
import re
import tensorflow as tf

porter=PorterStemmer()

                
def fillDictionary(dic, listReview):
    for j in range (len(listReview)):
        review=listReview[j].split()
        for i in range(len(review)):
            if(dic.get(review[i])!=None):
                dic[review[i]]=dic.get(review[i])+1
            else:
                dic[review[i]]=1
    return dic

def createDefAppendAllReview(listAllReview, listReview):
    for review in listReview:
        listAllReview.append(review)
    return listAllReview

def fillX(xDataFrame, listAllReviews):
    for i in range(len(listAllReviews)):
        listWords=listAllReviews[i].split()
        for word in listWords:
            #print(xDataFrame.at[word,xDataFrame[i]])
            xDataFrame[word][i]+=1
    return xDataFrame 

def replacePonctuation(review):
    review=re.sub("[^a-zA-Z]"," ",review)
    return review

def reviewList(listFile):
    listReview=[]
    for row in listFile:
        listReview.append(str(replacePonctuation(row[7])).lower())
    return listReview    

def deleteStopWords(listReview, listStopWords):
    newListReview=[]
    for row in listReview:
        review=row.split()
        review=[word for word in review if word.lower() not in listStopWords]
        newListReview.append(" ".join(review))
    return newListReview


def modifyWordPorter(listReviewOrig):
    listReview=listReviewOrig
    for j in range (len(listReview)):
        review=listReview[j].split()
        for i in range(len(review)):
            review[i]=porter.stem(review[i])
        listReview[j]=" ".join(review)
    return listReview
        
listAlways=[]
listStopWords=[]
listGillette=[]
listOralb=[]
listPantene=[]
listTampax=[]
listAllDocument=[]


#Ouverture des fichiers
fileStopWords=open('stop_words.txt','r')
fileAlways=open('datasets_clean/reviews_always.csv','r')
fileAlwayscsv=csv.reader(fileAlways)
fileGillette=open('datasets_clean/reviews_gillette.csv','r')
fileGillettecsv=csv.reader(fileGillette)
fileOralb=open('datasets_clean/reviews_oral-b.csv','r')
fileOralbcsv=csv.reader(fileOralb)
filePantene=open('datasets_clean/reviews_pantene.csv','r')
filePantenecsv=csv.reader(filePantene)
fileTampax=open('datasets_clean/reviews_tampax.csv','r')
fileTampaxcsv=csv.reader(fileTampax)

#Contenu des fichiers en liste
for row in fileStopWords:
    listStopWords.append(str(row.rstrip('\n')).lower())

for row in fileAlwayscsv:
    listAlways.append(row) 

for row in fileGillettecsv:
    listGillette.append(row) 
 
for row in fileOralbcsv:
    listOralb.append(row) 

for row in filePantenecsv:
    listPantene.append(row)
  
for row in fileTampaxcsv:
    listTampax.append(row)

listAlways.pop(0)
listGillette.pop(0)
listOralb.pop(0)
listPantene.pop(0)
listTampax.pop(0)

#Mettre les reviews dans une liste
listAlwaysReview=reviewList(listAlways)
listGilletteReview=reviewList(listGillette)
listOralbReview=reviewList(listOralb)
listPanteneReview=reviewList(listPantene)
listTampaxReview=reviewList(listTampax)

    
#Delete stop words
newListAlwaysReview=deleteStopWords(listAlwaysReview, listStopWords)
newListGilletteReview=deleteStopWords(listGilletteReview, listStopWords)
newListOralbReview=deleteStopWords(listOralbReview, listStopWords)
newListPanteneReview=deleteStopWords(listPanteneReview, listStopWords)
newListTampaxReview=deleteStopWords(listTampaxReview, listStopWords)

    

####################################################################################
############################### STEMMING PORTER ####################################
####################################################################################

listAlwaysReviewStemming=modifyWordPorter(newListAlwaysReview)
listGilletteReviewStemming=modifyWordPorter(newListGilletteReview)
listOralbReviewStemming=modifyWordPorter(newListOralbReview)
listPanteneReviewStemming=modifyWordPorter(newListPanteneReview)
listTampaxReviewStemming=modifyWordPorter(newListTampaxReview)

####################################################################################
############################# Bag representation ###################################
####################################################################################

#dic=OrderedDict()
dic={}
dic=fillDictionary(dic, listAlwaysReviewStemming)
#dic=fillDictionary(dic, listGilletteReviewStemming)
#dic=fillDictionary(dic, listOralbReviewStemming)
#dic=fillDictionary(dic, listPanteneReviewStemming)
#dic=fillDictionary(dic, listTampaxReviewStemming)
listWords=dic.keys()

listAllDocument=createDefAppendAllReview(listAllDocument, listAlwaysReviewStemming)
#listAllDocument=createDefAppendAllReview(listAllDocument, listGilletteReviewStemming)
#listAllDocument=createDefAppendAllReview(listAllDocument, listOralbReviewStemming)
#listAllDocument=createDefAppendAllReview(listAllDocument, listPanteneReviewStemming)
#listAllDocument=createDefAppendAllReview(listAllDocument, listTampaxReviewStemming)
#print(listAllDocument)

xDataFrame=pandas.DataFrame(index=range(len(listAllDocument)), columns=listWords)
xDataFrame=xDataFrame.fillna(0)
#print(xDataFrame)
#print(dic)

matrixX=fillX(xDataFrame, listAllDocument)
print(matrixX)

yDataFrame=pandas.DataFrame(index=range(len(listAllDocument)), columns=range(5))
matrixY=yDataFrame.fillna(0)