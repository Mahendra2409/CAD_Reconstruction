import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set the path to your main directory containing the 100 folders
main_dir = r"P:\CAD_Retrival\a_c_Filtered_2_C_Clusters\Set_4_output"

# Get a sorted list of folder names within the main directory
folders = sorted([f for f in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, f))])
folder_index = 0  # Start with the first folder

def load_images(folder_path):
    """Load and return all images from a given folder."""
    images = []
    for file in sorted(os.listdir(folder_path)):    
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(folder_path, file)
            img = cv2.imread(file_path)
            if img is not None:
                # Convert from BGR to RGB for proper color display in Matplotlib
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
    return images

def display_images(images, cols=4):
    """
    Clear the current figure and display the list of images in a grid.
    Reuses the same figure so that previous images are removed.
    """
    fig = plt.gcf()  # Get the current figure
    fig.clf()         # Clear the figure

    if not images:
        print("No images found in this folder.")
        return

    # Determine the grid layout: calculate the number of rows needed
    rows = (len(images) + cols - 1) // cols

    # Create subplots in the current figure (do not pass figsize here)
    axes = fig.subplots(rows, cols)
    # Ensure axes is a flat array for easy iteration
    axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else [axes]

    # Plot each image
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")

    # Hide any unused subplots
    for ax in axes[len(images):]:
        ax.axis("off")

    fig.tight_layout()
    fig.canvas.draw()  # Update the figure canvas

def update_display():
    """Load images from the current folder and update the display."""
    global folder_index
    folder_path = os.path.join(main_dir, folders[folder_index])
    images = load_images(folder_path)
    display_images(images)

def on_key(event):
    """Event handler to catch key presses and update folder display."""
    global folder_index
    if event.key == "right":
        folder_index = (folder_index + 1) % len(folders)  # Move to next folder (wrap around if needed)
        update_display()

# Create a global figure with the desired size
fig = plt.figure(figsize=(12, 8))
fig.canvas.mpl_connect("key_press_event", on_key)

# Display the images from the first folder
update_display()

# Start the Matplotlib event loop
plt.show()
