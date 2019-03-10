import requests
import json
import urllib.parse

##################################
# Modify only those 2 parameters #
##################################

nr_images = '10'  # if you want to download all the images = 23906
offset_imgs = '0' # if you want to start downloading from a specific image
nr_imgs_in_zip = 3  # cannot be higher than 300


url = 'https://isic-archive.com/api/v1/'
get_images_and_metadata_method = 'image?limit=' + nr_images + '&offset=' + offset_imgs + '&sort=name&sortdir=1&detail=true'

# complete URL to get all the images ID and the tag benign/malignant
url_request = url + get_images_and_metadata_method

# get a json containing the information about each image
response = requests.get(url_request)
img_data = response.json()

# data that is collected from each image
img_ids = []
all_data = []

##########################################
# write metadata on a separate json file #
##########################################

def getImageClassificationTag(image):

    try:
        imageClassification = image['meta']['clinical']['benign_malignant']

        if (imageClassification is None):
            return "None"
        else:
            return imageClassification
			
    except Exception as e:
	
        print("Cannot extract the classification of image " + str(image['name']) + ":" + str(e))
        return "_Fetch_Error_"

# loop over all the images and store the ids of the images, the name of the file and the tags (benign/malignant)
for img in img_data:
    id = img['_id']
    name = img['name']
    classification_tag = getImageClassificationTag(img)

    dict_object = dict(_id=id, name=name, benign_malignant=classification_tag)
    all_data.append(dict_object)
    img_ids.append(id)

# get a file object with write permission
file_object = open('metadata.json', 'w')

# store the information into a json file
# Save dict data into the JSON file.
json.dump(all_data, file_object)
print('metadata.json created.')

# close the object of the file
file_object.close()


###########################
# download all the images #
###########################

# creates chunks containing N image's ids
chunks = [img_ids[x:x+nr_imgs_in_zip] for x in range(0, len(img_ids), nr_imgs_in_zip)]

# creates zips containing N images
for i in range(0, len(chunks)):
    # convert the ids of the images into json string then into URL convention
    json_string = json.dumps(chunks[i])
    ids_str_url = urllib.parse.quote_plus(json_string)

    # request to the API
    url_request_download = url + 'image/download?include=images&imageIds=' + ids_str_url
    response_download = requests.get(url_request_download)

    # creation of the zip
    with open('img_' + "{:02d}".format(i) + '.zip', 'wb') as f:
        f.write(response_download.content)
        print('img_' + "{:02d}".format(i) + '.zip created.')
    f.close()
