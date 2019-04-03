# FS2019_ATML_Group2
Project Repo of Group 2 - Advanced Topics in Machine Learning FS2019 - Dataset downloader

## Description

This folder contains a script that downloads the images and the necessary metadata of melanoma's cases needed for a classification problem from the [ISIC gallery](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery). 
It also contains a script for reading metadata and another for extracting the downloaded zip files.

## Prerequisites

To run properly, the script needs to import the following packages:
- requests
- json
- urllib.parse

## Execution

Before running the program define the number of image you want to download (max 23906) and the number of images contained in each zip (max 300).

Example:
```python
nr_images = '10'
nr_imgs_in_zip = 3
```

To execute the downloader, type on the command line the following: 
```python
python data_downloader.py
```

## Data extraction

Once all the zip files have been downloaded, the extractor script can be launched with:
```python
python data_extractor.py
```

The script will extract the zip file, then move all images in the "../ISIC-images/benign" respectively "../ISIC-images/malignant/"
folders. It will also make a copy of the metadata file containing only the data of the images that could be successfuly extracted.

Before running the program, you can choose if you want to extract all the zip files or only the first three. 
You can also choose whether you want to keep the zip files or delete them after the extraction. Same goes for 
the intermediate folders created during the extraction, you can choose to keep them or delete them.

Example:
```python
extractAllZipFiles = False
deleteZipFiles = False
deleteIntermediateFolder = True
```

Remark:
Some of the data don't have a benign or malignant classification. These are not moved and stay in the intermediate folders
created during extraction.