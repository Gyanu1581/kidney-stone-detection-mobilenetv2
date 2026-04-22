import os

DATA_DIR = "dataset_raw/Original_Dataset"

print(f"\n🔍 Looking inside: {os.path.abspath(DATA_DIR)}")

if not os.path.exists(DATA_DIR):
    print("❌ Folder does not exist!")
else:
    for folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder)
        if os.path.isdir(folder_path):
            file_count = len(os.listdir(folder_path))
            print(f"📁 Found folder '{folder}' with {file_count} images.")