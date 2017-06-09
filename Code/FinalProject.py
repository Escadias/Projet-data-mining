#!/usr/bin/python

import csv
from nltk.stem import PorterStemmer


def openFile(fileName):
    if fileName == 'stop_words.txt':
        return open(fileName, 'r')
    else:
        return open('datasets_clean/' + fileName, 'r')

def loadFileRows(file, isStopWords):
    rowsList = []
    if isStopWords:
        for row in file:
            rowsList.append(str(row.rstrip('\n')).upper())
    else:
        for row in file:
            rowsList.append(row)
        rowsList.pop(0)

    return rowsList

def loadReviews(rowsList):
    reviewList = []
    for row in rowsList:
        if(row[7].find('\\')):
            row[7]=row[7].replace('\\','')
        if(row[7].find('\'')):
            row[7]=row[7].replace('\'','')
        if(row[7].find('.')):
            row[7]=row[7].replace('.','')
        if(row[7].find('!')):
            row[7]=row[7].replace('!','')
        if(row[7].find(':')):
            row[7]=row[7].replace(':','')
        if(row[7].find('-')):
            row[7]=row[7].replace('-','')
        if(row[7].find('?')):
            row[7]=row[7].replace('?','')
        reviewList.append(str(row[7]).upper())

    return reviewList

def deleteStopWordsInReviewList(reviewsList, stopWordsList):
    noStopWordsReviewsList=[]
    for row in reviewsList:
        reviewsList=row.split()
        reviewsList=[word for word in reviewsList if word.upper() not in stopWordsList]
        noStopWordsReviewsList.append(" ".join(reviewsList))
    return noStopWordsReviewsList


listAlways=[]
listStopWords=[]
listGillette=[]
listOralb=[]
listPantene=[]
listTampax=[]

listAlwaysReview=[]
listGilletteReview=[]
listOralbReview=[]
listPanteneReview=[]
listTampaxReview=[]

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
    listStopWords.append(str(row.rstrip('\n')).upper())

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
for row in listAlways:
    if(row[7].find('\\')):
        row[7]=row[7].replace('\\','')
    if(row[7].find('\'')):
        row[7]=row[7].replace('\'','')
    if(row[7].find('.')):
        row[7]=row[7].replace('.','')
    if(row[7].find('!')):
        row[7]=row[7].replace('!','')
    if(row[7].find(':')):
        row[7]=row[7].replace(':','')
    if(row[7].find('-')):
        row[7]=row[7].replace('-','')
    if(row[7].find('?')):
        row[7]=row[7].replace('?','')
    listAlwaysReview.append(str(row[7]).upper()) 

for row in listGillette:
    if(row[7].find('\\')):
        row[7]=row[7].replace('\\','')
    if(row[7].find('\'')):
        row[7]=row[7].replace('\'','')
    listGilletteReview.append(str(row[7]).upper()) 
    
for row in listOralb:
    if(row[7].find('\\')):
        row[7]=row[7].replace('\\','')
    if(row[7].find('\'')):
        row[7]=row[7].replace('\'','')
    listOralbReview.append(str(row[7]).upper()) 

for row in listPantene:
    if(row[7].find('\\')):
        row[7]=row[7].replace('\\','')
    if(row[7].find('\'')):
        row[7]=row[7].replace('\'','')
    listPanteneReview.append(str(row[7]).upper()) 

for row in listTampax:
    if(row[7].find('\\')):
        row[7]=row[7].replace('\\','')
    if(row[7].find('\'')):
        row[7]=row[7].replace('\'','')
    listTampaxReview.append(str(row[7]).upper())
    
#Delete stop words
newListAlwaysReview=[]
for row in listAlwaysReview:
    reviewAlways=row.split()
    reviewAlways=[word for word in reviewAlways if word.upper() not in listStopWords]
    newListAlwaysReview.append(" ".join(reviewAlways))
    
newListGilletteReview=[]
for row in listGilletteReview:
    reviewGillette=row.split()
    reviewGillette=[word for word in reviewGillette if word.upper() not in listStopWords]
    newListGilletteReview.append(" ".join(reviewGillette))

newListOralbReview=[]
for row in listOralbReview:
    reviewOralb=row.split()
    reviewOralb=[word for word in reviewOralb if word.upper() not in listStopWords]
    newListOralbReview.append(" ".join(reviewOralb))
    
newListPanteneReview=[]
for row in listPanteneReview:
    reviewPantene=row.split()
    reviewPantene=[word for word in reviewPantene if word.upper() not in listStopWords]
    newListPanteneReview.append(" ".join(reviewPantene))
    
newListTampaxReview=[]
for row in listTampaxReview:
    reviewTampax=row.split()
    reviewTampax=[word for word in reviewTampax if word.upper() not in listStopWords]
    newListTampaxReview.append(" ".join(reviewTampax))
    
#print(newListAlwaysReview)

####################################################################################
############################### STEMMING PORTER ####################################
####################################################################################

porter=PorterStemmer()

for j in range (len(newListAlwaysReview)):
    reviewAlways=newListAlwaysReview[j].split()
    for i in range(len(reviewAlways)):
        reviewAlways[i]=porter.stem(reviewAlways[i])
    newListAlwaysReview[j]=" ".join(reviewAlways)
    
#print(newListAlwaysReview)


dic={}
#dic['a']=1
#dic['a']=dic.get('a')+255
#print(dic['a'])

for j in range (len(newListAlwaysReview)):
    reviewAlways=newListAlwaysReview[j].split()
    for i in range(len(reviewAlways)):
        if(dic.get(reviewAlways[i])!=None):
            dic[reviewAlways[i]]=dic.get(reviewAlways[i])+1
        else:
            dic[reviewAlways[i]]=1

print(dic.get('buy'))