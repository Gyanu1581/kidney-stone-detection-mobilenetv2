import kagglehub
import shutil
import os

# Download dataset
path = kagglehub.dataset_download(
    "orvile/axial-ct-imaging-dataset-kidney-stone-detection"
)

print("Downloaded at:", path)

# OPTIONAL: move dataset to project dataset folder
DEST = "../dataset_raw"

if not os.path.exists(DEST):
    shutil.copytree(path, DEST)
    print("Dataset copied to dataset_raw/")
else:
    print("dataset_raw already exists")