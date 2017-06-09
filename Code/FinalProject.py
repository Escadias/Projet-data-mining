#!/usr/bin/python

import csv
from nltk.stem import PorterStemmer

porter=PorterStemmer()

def replacePonctuation(review):
    if(review.find('.')):
        review=review.replace('.',' ')
    if(review.find(',')):
        review=review.replace(',',' ')
    if(review.find('!')):
        review=review.replace('!',' ')
    if(review.find('?')):
        review=review.replace('?',' ')
    if(review.find(':')):
        review=review.replace(':',' ')
    if(review.find('...')):
        review=review.replace('...',' ')
    if(review.find(r"\'")):
        review=review.replace(r"\'",'')
    if(review.find('-')):
        review=review.replace('-',' ')
    if(review.find('\\')):
        review=review.replace('\\',' ')
    if(review.find('$')):
        review=review.replace('$','')
    if(review.find('&')):
        review=review.replace('&','')
    if(review.find('(')):
        review=review.replace('(','')
    if(review.find(')')):
        review=review.replace(')','')
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


def modifyWordPorter(listReview):
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

dic={}

for j in range (len(newListAlwaysReview)):
    reviewAlways=newListAlwaysReview[j].split()
    for i in range(len(reviewAlways)):
        if(dic.get(reviewAlways[i])!=None):
            dic[reviewAlways[i]]=dic.get(reviewAlways[i])+1
        else:
            dic[reviewAlways[i]]=1

