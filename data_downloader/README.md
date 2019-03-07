# FS2019_ATML_Group2
Project Repo of Group 2 - Advanced Topics in Machine Learning FS2019 - Dataset downloader

## Description

This folder contains a script that downloads the images and the necessary metadata of melanoma's cases needed for a classification problem from the [ISIC gallery](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery). 

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
