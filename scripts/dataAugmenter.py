from PIL import Image
from torchvision import transforms
import glob, os

# Resize all images in the given directory
def resizeData(dir):
    
    print('Resizing images in ' + dir + '...')
        
    #TODO: keep the aspect ratio
    resize = transforms.Resize((300,300))
        
    for subDir in ['benign/', 'malignant/']:
        if not os.path.exists(dir+subDir):
            print("Folder doesn't exist, try splitting the data again")
        else:
            imgPathList = glob.glob(dir + subDir + '*.jpg')
            for imgPath in imgPathList:
                img = Image.open(imgPath)
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
def preprocessData(directories):
    
    for directory in directories:
        resizeData(directory)
        augmentMalignant(directory)    