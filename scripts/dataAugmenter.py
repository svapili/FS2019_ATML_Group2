from PIL import Image
from torchvision import transforms
import glob, os

'''
Resize all images in the given directory
dir:    the path to the train, test or val directories
size:   a tuple holding the dimensions of the resized image. E.g. (300,200)
keepAspectRatio: a boolean saying if the image aspect ratio has to be kept. In
                 that case, black padding are added to the image.
'''
def resizeData(path, outSize=(300,300), keepAspectRatio=False):
    
    print('Resizing images in ' + path + '...')
        
    for subDir in ['benign/', 'malignant/']:
        if not os.path.exists(path+subDir):
            print("Folder doesn't exist. Try to do a new data split.")
        else:
            imgPathList = glob.glob(path + subDir + '*.jpg')
            for imgPath in imgPathList:
                img = Image.open(imgPath)
                if (keepAspectRatio):
                    inSize = img.size
                    ratio = float(outSize[0])/max(inSize)
                    newSize = tuple([int(x*ratio) for x in inSize])
                    resize = transforms.Resize((newSize[1],newSize[0]))
                    resizedImg = resize(img)
                    newImg = Image.new("RGB", outSize)
                    newImg.paste(resizedImg, ((outSize[0]-newSize[0])//2, (outSize[1]-newSize[1])//2))
                    newImg.save(imgPath)
                else:
                    resize = transforms.Resize(outSize)
                    resizedImg = resize(img)
                    resizedImg.save(imgPath)

    print('Done!')           

                
# Augment malignant images of the given directory
def augmentMalignant(dir):
    
    subDir = 'malignant/'
    
    print('Augmenting malignant class...')
    
    augmentations = [transforms.RandomHorizontalFlip(p=1.0),
                     transforms.RandomVerticalFlip(p=1.0),
                     transforms.Compose([
                             transforms.RandomHorizontalFlip(p=1.0),
                             transforms.RandomVerticalFlip(p=1.0)
                             ])
                    ]   

    if not os.path.exists(dir+subDir):
        print("Folder doesn't exist, try splitting the data again")
    else:
        
        # Clear previously augmented data
        imgPathList = glob.glob(dir + subDir + '*_[0-9].jpg')
        if imgPathList:
            for imgPath in imgPathList:
                os.remove(imgPath)
        
        # Augment data
        imgPathList = glob.glob(dir + subDir + '*.jpg')
        for imgPath in imgPathList:
            file, ext = os.path.splitext(imgPath)
            img = Image.open(imgPath)
            for index, augmentation in enumerate(augmentations):
                augmentedImg = augmentation(img)
                destination = file + '_' + str(index) + ext
                augmentedImg.save(destination)        
        print('Done!')
  

# Erase augmented images, i.e. images ending in '*_[0-9].jpg'     
def clearAugmented(dir):
    
    subDir = 'malignant/'
    
    imgPathList = glob.glob(dir + subDir + '*_[0-9].jpg')
    
    if imgPathList:
        for imgPath in imgPathList:
            os.remove(imgPath)


# Resize and augment data from the given directories
def preprocessData(directories, outSize=(300,300), keepAspectRatio=False):
   
    for directory in directories:
        resizeData(directory, outSize, keepAspectRatio)