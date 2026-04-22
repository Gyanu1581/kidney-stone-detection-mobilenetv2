import os
import shutil
import kagglehub

DEST = "../dataset_raw"

def reset_dataset():
    # 1. Delete the existing, messy dataset_raw folder
    if os.path.exists(DEST):
        print(f"🗑️  Deleting existing contaminated '{DEST}' folder...")
        shutil.rmtree(DEST)
        print("✅  Deletion complete.")

    # 2. Download a fresh copy from Kaggle
    print("\n⏳ Downloading fresh dataset from Kaggle...")
    path = kagglehub.dataset_download("orvile/axial-ct-imaging-dataset-kidney-stone-detection")
    print(f"✅  Download complete. Cached at: {path}")

    # 3. Locate the "Original" folder inside the downloaded cache
    # Handling potential naming variations from the Kaggle uploader
    original_path_1 = os.path.join(path, "Original_Dataset")
    original_path_2 = os.path.join(path, "Original Dataset") 
    
    clean_source = None
    if os.path.exists(original_path_1):
        clean_source = original_path_1
    elif os.path.exists(original_path_2):
        clean_source = original_path_2
    else:
        print("⚠️ Could not find the specific 'Original' folder. Copying the entire root.")
        clean_source = path

    # 4. Copy ONLY the pristine data into your project
    print(f"\n📦 Extracting clean images to '{DEST}'...")
    shutil.copytree(clean_source, DEST)
    
    # Verify the final structure
    folders = os.listdir(DEST)
    print(f"✅  Data reset complete! '{DEST}' now strictly contains: {folders}")

if __name__ == "__main__":
    reset_dataset()