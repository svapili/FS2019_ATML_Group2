import numpy as np
import requests
import json

nr_images = '5'  # if you want to download all the images = 23906
url = 'https://isic-archive.com/api/v1/'
get_images_and_metadata_method = 'image?limit=' + nr_images + '&sort=name&sortdir=1&detail=true'

# complete URL to get all the images ID and the tag benign/malignant
url_request = url + get_images_and_metadata_method

# get a json containing the information about each image
response = requests.get(url_request)
img_data = response.json()

# data that is collected from each image
img_ids = []
img_name = []
img_classification_tags = []
all_data = []

# loop over all the images and store the ids of the images, the name of the file and the tags (benign/malignant)
for img in img_data:
    id = img['_id']
    name = img['name']
    classification_tag = img['meta']['clinical']['benign_malignant']

    dict_object = dict(_id=id, name=name, benign_malignant=classification_tag)
    all_data.append(dict_object)
    
    
###########################
# download all the images #
###########################

for img in img_ids:
    id = img['_id']
    url_request_download = url + 'image/' + id + '/download'
    response_download = requests.get(url_request_download)
    
##########################################
# write metadata on a separate json file #
##########################################
    
# Get a file object with write permission.
file_object = open('metadata.json', 'w')

# store the information into a json file
try:
    # Save dict data into the JSON file.
    json.dump(all_data, file_object)

    print('metadata.json' + " created. ")    
except FileNotFoundError:
    print('metadata.json' + " not found. ") 
    
# close the object of the file
file_object.close()