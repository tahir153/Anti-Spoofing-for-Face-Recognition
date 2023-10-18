import os
import shutil
import random
from itertools import islice

outputFolderPath = "Dataset/SplitData"
inputFolderPath = "Dataset/All"
splitRatio = {"train":0.7,"val":0.2,"test":0.1}
classes = ["fake","real"]

try:
    shutil.rmtree(outputFolderPath)
except OSError as e:
    os.mkdir(outputFolderPath)
#-------------- Making Directories -----------
os.makedirs(f"{outputFolderPath}/train/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels",exist_ok=True)
#-------------- Get the names -----------
listNames = os.listdir(inputFolderPath)
uniqueNames = []
for name in listNames:
    uniqueNames.append(name.split(".")[0])
uniqueNames = list(set(uniqueNames))
#-------------- shuffle -----------
random.shuffle(uniqueNames)
#-------------- find the number of images for each folder -----------
lenData = len(uniqueNames)
lenTrain = int(lenData*splitRatio['train'])
lenVal = int(lenData*splitRatio['val'])
lenTest = int(lenData*splitRatio['test'])

#-------------- Put the remaining images to training -----------
if lenData != lenTrain + lenTest + lenVal:
    remaining = lenData-(lenTrain+lenVal+lenTest)
    lenTrain += remaining

#-------------- split the list -----------
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for  elem in lengthToSplit]
# print(Output)
print(f'Total Images:{lenData}  Split: {len(Output[0])} {len(Output[1])} {len(Output[2])}')
#-------------- Copy the files -----------
sequence = ['train','val', 'test']
for i, out in enumerate(Output):
    for filename in out:
        shutil.copy(f'{inputFolderPath}/{filename}.jpg', f'{outputFolderPath}/{sequence[i]}/images/{filename}.jpg')
        shutil.copy(f'{inputFolderPath}/{filename}.txt', f'{outputFolderPath}/{sequence[i]}/labels/{filename}.txt')
print('Spliting Process Completed!')

#-------------- Creating data.yaml file -----------
dataYaml = f'path: ../Data\n\
train: ../train/image\n\
val: ../val/image\n\
test: ../test/image\n\
\n\
nc: {len(classes)} \n\
names: {classes}'
f = open(f'{outputFolderPath}/data.yaml',"a")
f.write(dataYaml)
f.close()



print('Spliting Process Completed!')



