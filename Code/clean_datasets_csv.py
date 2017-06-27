import os
from os import listdir, makedirs
from os.path import isfile, join
datasetsPath = "datasets/"
datasetsCleanPath = "datasets_clean/"

print(datasetsCleanPath)

os.makedirs(os.path.dirname(datasetsCleanPath), exist_ok=True)
listFiles = [f for f in os.listdir(datasetsPath) if os.path.isfile(os.path.join(datasetsPath, f))]
print(listFiles)


for fileName in listFiles:
	file = open(datasetsPath + fileName, "r")
	fileClean = open(datasetsCleanPath + fileName, "w")

	lineCounter = 0
	for line in file:
		if line[0].isdigit() or lineCounter == 0 :
			fileClean.write(line)
		#print(line[0:7],"\t",lineCounter,"\t",fileName)
		lineCounter+=1

	file.close()
	fileClean.close()