'''
File containing all necessary functions for splitting the data into test and
validation sets from the training set.

At the beginning the data is expected to be gathered in the training directory
with the following structure (as created by data_extractor.py):
    
    trainDir/
        |_ benign/
        |_ malignant/ 
        
Or the data can also be already split in the final folder structure:
    
    trainDir/
        |_ benign/
        |_ malignant/
    testDir/
        |_ benign/
        |_ malignant/ 
    valDir/
        |_ benign/
        |_ malignant/
        
The entry point of this script is the function split(trainDir, testDir, valDir, 
testRatio, valRatio)
'''

import os, os.path
import glob
import numpy as np
import shutil
import platform

if platform.platform()[0:5] == 'Linux':
    Linux = True
else:
    Linux = False


'''
Create directory structure and puts back all images in the train folder if they
were already split.
trainDir:   the path to the train directory
testDir:    the path to the test directory
valDir:     the path to the validation directory
'''
def initDir(trainDir, testDir, valDir):
    
    newPathList = []
    newPathList.append(trainDir)
    newPathList.append(testDir)
    newPathList.append(valDir)
    
    for newPath in newPathList:
        # Creates missing directories...
        if not os.path.exists(newPath):
            os.makedirs(newPath)
            for subDir in ['benign/', 'malignant/']:
                if not os.path.exists(newPath+subDir):
                    os.makedirs(newPath+subDir)
        # ...or gather back all images in the train directory if already split
        else:
            for subDir in ['benign/', 'malignant/']:
                if not os.path.exists(newPath+subDir):
                    os.makedirs(newPath+subDir) 
                else:
                    imgList = glob.glob(newPath + subDir + '*.jpg')
                    for img in imgList:
                        src = img
                        if Linux:
                            dest = newPath.rsplit('/',2)[0] + '/train/' + subDir + img.split('/')[-1]
                        else:
                            dest = newPath.rsplit('/',2)[0] + '/train/' + subDir + img.split('\\')[-1]

                        shutil.move(src, dest)


'''
Print image count in each of the directories
directories: a list containing the path to the train, test and val directories
'''
def printImageCnt(directories):
    for directory in directories:
        benignCnt = len(getImageList(directory+'benign/'))
        malignCnt = len(getImageList(directory+'malignant/'))
        print(directory)
        print("Benign images: " + str(benignCnt))
        print("Malignant images: " + str(malignCnt)+"\n")   


'''
Get a list of the paths to every image in the directory
directory: the directory of interest
'''                
def getImageList(directory):
    imageList = glob.glob(directory + '*jpg')
    return imageList


'''
Get the list of img indexes, shuffle them and split them with the given ratio
Args
setLength:  the number of elements in the image set
ratio:      the split ratio
randomSeed: a random seed for the random number generator
Return
firstSubsetIndices:  an index list of the images in the first subset
secondSubsetIndices: an index list of the images in the second subset
'''
def getSplitIndices(setLength, ratio, randomSeed):
    
    indices = list(range(setLength))
    split = int(np.floor(ratio * setLength))
    
    np.random.seed(randomSeed)
    np.random.shuffle(indices)
    
    firstSubsetIndices = indices[split:]
    secondSubsetIndices = indices[:split]
    
    return firstSubsetIndices, secondSubsetIndices


'''
From a given image list, move the images with the given indices to a given
directory
moveDir: the destination folder in which to move the images
indices: the indices of the images to move
imgList: a list of the paths to the source images
'''     
# 
def moveImages(moveDir, indices, imgList):
    for idx in indices:
        src = imgList[idx]
        if Linux:
            dest = moveDir + imgList[idx].split('/')[-1]
        else:
            dest = moveDir + imgList[idx].split('\\')[-1]

        shutil.move(src, dest)
 
       
'''
Split the train set into a train and test sets according to a given ratio
trainDir:   the path to the train directory
testDir:    the path to the test directory
testRatio:  the split ratio
randomSeed: a random seed for the random number generator
'''
def splitTrainTest(trainDir, testDir, testRatio, randomSeed):
    benignList = getImageList(trainDir+'benign/')
    malignList = getImageList(trainDir+'malignant/')
    
    benignLength = len(benignList)
    malignLength = len(malignList)
    
    benignTrainIndices, benignTestIndices = getSplitIndices(benignLength, 
                                                            testRatio, 
                                                            randomSeed)
    malignTrainIndices, malignTestIndices = getSplitIndices(malignLength, 
                                                            testRatio, 
                                                            randomSeed)
    
    moveImages(testDir+'benign/', benignTestIndices, benignList)
    moveImages(testDir+'malignant/', malignTestIndices, malignList)
 
    
'''
Split the train set into train and validation sets according to a given ratio
trainDir:   the path to the train directory
valDir:     the path to the validation directory
valRatio:   the split ratio
randomSeed: a random seed for the random number generator
'''
def splitTrainVal(trainDir, valDir, valRatio, randomSeed):
    benignList = getImageList(trainDir+'benign/')
    malignList = getImageList(trainDir+'malignant/')
    
    benignLength = len(benignList)
    malignLength = len(malignList)
    
    benignTrainIndices, benignValIndices = getSplitIndices(benignLength, 
                                                           valRatio, 
                                                           randomSeed)
    malignTrainIndices, malignValIndices = getSplitIndices(malignLength, 
                                                           valRatio, 
                                                           randomSeed)
    
    moveImages(valDir+'benign/', benignValIndices, benignList)
    moveImages(valDir+'malignant/', malignValIndices, malignList) 


'''
Split the data into train, test and validation set according to the given 
ratios
trainDir:   the path to the train directory
testDir:    the path to the test directory
valDir:     the path to the validation directory
testRatio:  the ratio of test images
valRatio:   the ratio of validation images
'''
def split(trainDir, testDir, valDir, testRatio, valRatio):
    
    # Variables
    randomSeed= 42
      
    print('Initializing directories...\n')         
    initDir(trainDir, testDir, valDir)
    
    print("Initial repartition:\n")
    printImageCnt([trainDir, testDir, valDir])

    print("Creating test set...")
    splitTrainTest(trainDir, testDir, testRatio, randomSeed)
    
    print("Creating validation set...")
    splitTrainVal(trainDir, valDir, valRatio, randomSeed)
    
    print("Job done!\n")
    print("Image repartition: \n")  
    printImageCnt([trainDir, testDir, valDir])
