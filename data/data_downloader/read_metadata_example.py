# This script shows how to access the ISIC metadata.json
# run "python read_metadata_example.py"

import os
import json
from pprint import pprint

print("Current Working Directory " , os.getcwd())

with open('metadata.json') as f:
    data = json.load(f)

pprint(data)
