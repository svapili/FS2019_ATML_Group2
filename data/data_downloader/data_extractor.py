import glob
import zipfile
import shutil
from os import remove

# Variables
deleteZipFiles = False

# Directories
zip_file_dir = './'
extract_dir = '../'
move_dir = '../ISIC-images/'

#####################
# Extract zip files #
#####################
print('Unzipping files...')

zip_file_list = glob.glob(zip_file_dir + 'img*.zip')

for zip_file in zip_file_list:
    print(zip_file)
    zip_ref = zipfile.ZipFile(zip_file, 'r')
    zip_ref.extractall(extract_dir)
    zip_ref.close()

######################################
# Move all images to the same folder #
######################################
print('Moving images...')

img_path_list = glob.glob(extract_dir + '/ISIC-images/**/*.jpg')

for img_path in img_path_list:
    img_name = img_path.split('\\')[-1]
    shutil.move(img_path, move_dir+img_name)

################################
# Copy metadata to extract_dir #
################################
print('Copying metadata...')

shutil.copy(zip_file_dir + 'metadata.json', extract_dir + 'metadata.json')

#################################
# Deleting intermediate folders #
#################################
print('Deleting intermediate folders...')

folder_list = glob.glob(extract_dir + '/ISIC-images/**/')

for folder in folder_list:
    shutil.rmtree(folder)


######################
# Deleting zip files #
######################
if (deleteZipFiles):
    print('Deleting zip files...')

    for zip_file in zip_file_list:
        remove(zip_file)

print('Job done!')
