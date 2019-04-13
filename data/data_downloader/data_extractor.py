import glob
import zipfile
import shutil
import sys
import json
import os
from os import remove

# Variables
extractAllZipFiles = True # Set True to extract all zip files. Set False to extract only the first 3 zip files.
deleteZipFiles = False # Set True to delete all zip files at the end
deleteIntermediateFolder = False# Set True to delete intermediate folders at the end

# Directories
zip_file_dir = './'
image_dir = '../ISIC-images/'
move_dir = '../ISIC-images/train/'
benign_dir = '../ISIC-images/train/benign/'
malignant_dir = '../ISIC-images/train/malignant/'


#######################################
# Create folders #
#######################################
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
if not os.path.exists(move_dir):
    os.makedirs(move_dir)    
if not os.path.exists(benign_dir):
    os.makedirs(benign_dir)
if not os.path.exists(malignant_dir):
    os.makedirs(malignant_dir)

#####################
# Extract zip files #
#####################
print('Unzipping files...')

zip_file_list = glob.glob(zip_file_dir + 'img*.zip')

if (extractAllZipFiles):
    for zip_file in zip_file_list:
        print(zip_file)
        try:
            zip_ref = zipfile.ZipFile(zip_file, 'r')
            zip_ref.extractall(zip_file_dir)
            zip_ref.close()
        except Exception as e:
            print("Error with " + zip_file + ": " + e.args[0])
else:
    for zip_file in zip_file_list[0:3]:
        print(zip_file)
        try:
            zip_ref = zipfile.ZipFile(zip_file, 'r')
            zip_ref.extractall(zip_file_dir)
            zip_ref.close()
        except Exception as e:
            print("Error with " + zip_file + ": " + e.args[0])

#####################################################################
# Move all images to benign or malignant folder and update metadata #
#####################################################################
print('Moving images...')

img_path_list = glob.glob(zip_file_dir + '/ISIC-images/**/*.jpg')

with open(zip_file_dir + 'metadata.json') as f:
    metadata = json.load(f)
    
new_metadata = []

for img_path in img_path_list:
    
    img_name_format = img_path.split('\\')[-1]
    img_name = img_name_format.split('.')[0]
    
    for data in metadata:
        
        # Check if image in metadata exists
        if (data['name'] == img_name):
            new_metadata.append(data)
            
            # Move image to corresponding directory
            if (data['benign_malignant'] == 'benign'):
                shutil.move(img_path, benign_dir+img_name_format)
            elif (data['benign_malignant'] == 'malignant'):
                shutil.move(img_path, malignant_dir+img_name_format)
            else:
                print("---------")

# Create new metadata file
file_object = open(image_dir + 'metadata.json', 'w')
json.dump(new_metadata, file_object)
print('metadata.json created.')
file_object.close()


#################################
# Deleting intermediate folders #
#################################
if (deleteIntermediateFolder):
    print('Deleting intermediate folders...')
    
    shutil.rmtree(zip_file_dir + '/ISIC-images/')
    

######################
# Deleting zip files #
######################
if (deleteZipFiles):
    print('Deleting zip files...')

    for zip_file in zip_file_list:
        remove(zip_file)

print('Job done!')
