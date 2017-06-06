#!/usr/bin/python

import csv

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
fileAlways=open('datasets/reviews_always.csv','r')
fileAlwayscsv=csv.reader(fileAlways)
fileGillette=open('datasets/reviews_gillette.csv','r')
fileGillettecsv=csv.reader(fileGillette)
fileOralb=open('datasets/reviews_oral-b.csv','r')
fileOralbcsv=csv.reader(fileOralb)
filePantene=open('datasets/reviews_pantene.csv','r')
filePantenecsv=csv.reader(filePantene)
fileTampax=open('datasets/reviews_tampax.csv','r')
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
    listAlwaysReview.append(str(row[7]).upper()) 

for row in listGillette:
    listGilletteReview.append(str(row[7]).upper()) 
    
for row in listOralb:
    listOralbReview.append(str(row[7]).upper()) 

for row in listPantene:
    listPanteneReview.append(str(row[7]).upper()) 

for row in listTampax:
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
    
print(newListAlwaysReview)