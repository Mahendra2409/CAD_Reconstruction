import os
import hashlib
import shutil

# Paths
source_folder = "M:\Order to PC\CAD_Reconstruction\CAD_Retrival\Fusion360GalleryDataset\Data\Assembly\PNG1"  # Replace with your image folder path
output_folder = "M:\Order to PC\CAD_Reconstruction\CAD_Retrival\Fusion360GalleryDataset\Data\Assembly\PNG1(Removed Duplicates)"  # Folder where duplicate folders will be stored
distinct_folder = "M:\Order to PC\CAD_Reconstruction\CAD_Retrival\Fusion360GalleryDataset\Data\Assembly\PNG1(Distict)"  # Folder for unique images

# Ensure output folders exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(distinct_folder, exist_ok=True)

# Function to compute hash of an image
def compute_hash(image_path):
    try:
        hasher = hashlib.md5()
        with open(image_path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Dictionary to store image hashes and their original paths
hash_dict = {}
duplicates_dict = {}

# Scan folder for images
count = 0
for root, _, files in os.walk(source_folder):  # Use os.walk() to scan all files
    for filename in files:
        file_path = os.path.join(root, filename)
        
        if os.path.isfile(file_path):  # Ensure it's a file
            img_hash = compute_hash(file_path)
            
            if img_hash:
                if img_hash in hash_dict:
                    if img_hash not in duplicates_dict:
                        duplicates_dict[img_hash] = []  # Create list for duplicates

                    duplicates_dict[img_hash].append(file_path)
                else:
                    hash_dict[img_hash] = file_path  # Store first occurrence

        count += 1
        if count % 500 == 0:
            print(f"Processed {count} images...")  # Print progress every 500 images

# Move distinct images to Distinct_Objects folder
for img_hash, original_path in hash_dict.items():
    shutil.move(original_path, os.path.join(distinct_folder, os.path.basename(original_path)))

# Create folders and move duplicates
for img_hash, duplicate_paths in duplicates_dict.items():
    distinct_image_name = os.path.basename(hash_dict[img_hash]).split('.')[0]
    duplicate_folder = os.path.join(output_folder, distinct_image_name)
    
    os.makedirs(duplicate_folder, exist_ok=True)  # Create folder for each distinct image

    # Move duplicate images into the respective folder
    for dup in duplicate_paths:
        shutil.move(dup, os.path.join(duplicate_folder, os.path.basename(dup)))

print(f"\n Processed {count} images.")
print(f" Moved distinct images to {distinct_folder}.")
print(f" Moved duplicate images into separate folders in {output_folder}.")
