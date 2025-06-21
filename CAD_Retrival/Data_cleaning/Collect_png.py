import os
import shutil

# Set source and destination folders
source_directory = "Fusion360GalleryDataset\Data\Assembly\assembly"  # Change this to your main folder containing 1000 subfolders
destination_directory = "Fusion360GalleryDataset\Data\Assembly\PNG1"  # Change this to where you want to collect JPGs

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

# Loop through all folders and collect .jpg files
for root, _, files in os.walk(source_directory):
    
    for file in files:
        
        if file.lower().endswith(".png"):  # Check for JPG files (case insensitive)
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_directory, file)

            # If the file already exists, rename it to avoid overwriting
            counter = 1
            while os.path.exists(destination_path):
                name, ext = os.path.splitext(file)
                new_name = f"{name}_{counter}{ext}"
                destination_path = os.path.join(destination_directory, new_name)
                counter += 1

            # Copy the file
            shutil.copy2(source_path, destination_path)
            print(f"Copied: {source_path} -> {destination_path}")

print("All JPG files have been copied successfully!")
