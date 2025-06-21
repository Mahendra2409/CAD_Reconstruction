import os
import fitz
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import trimesh
import torch


def process_single_pdf(pdf_path, output_folder, dpi=600):
    """Process a single PDF file and extract its 8th page."""
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output_folder = os.path.join(output_folder, pdf_name)
    image_path = os.path.join(pdf_output_folder, f"{pdf_name}_Page_08.png")
    
    os.makedirs(pdf_output_folder, exist_ok=True)
    
    with fitz.open(pdf_path) as pdf_document:
        if len(pdf_document) > 7:
            page = pdf_document[8]
            image = page.get_pixmap(dpi=dpi)
            image.save(image_path)
            return pdf_name, image_path
    return None


def extract_eighth_page(pdf_folder, output_folder, dpi=600):
    """Sequential processing of PDFs."""
    pdf_files = [os.path.join(root, f) for root, _, files in os.walk(pdf_folder) 
                 for f in files if f.lower().endswith(".pdf")]
    
    image_paths = {}
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        result = process_single_pdf(pdf_file, output_folder, dpi)
        if result:
            pdf_name, path = result
            image_paths[pdf_name] = path
    
    return image_paths


def image_to_point_cloud(pdf_name, image_path, num_points, point_cloud_folder):
    """Convert image to point cloud."""
    obj_path = os.path.join(point_cloud_folder, f"{pdf_name}.obj")
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)
    edges = cv2.Canny(binary, 100, 200, apertureSize=3)
    
    y_coords, x_coords = np.where(edges > 0)
    coords = np.column_stack((y_coords, x_coords))
    
    if len(coords) > num_points:
        indices = np.random.choice(len(coords), num_points, replace=False)
        final_points = coords[indices]
    else:
        final_points = coords
    
    h = binary.shape[0]
    points = np.column_stack((final_points[:, 1], h - final_points[:, 0], np.ones(len(final_points))))
    np.savetxt(obj_path, points, fmt='v %.1f %.1f %.1f', delimiter=' ')
    
    return pdf_name, len(points)


def batch_process_point_clouds(image_paths, point_cloud_folder, num_points):
    """Sequential processing of point cloud conversion."""
    os.makedirs(point_cloud_folder, exist_ok=True)
    results = {}
    for pdf_name, image_path in tqdm(image_paths.items(), desc="Converting to point clouds"):
        results[pdf_name] = image_to_point_cloud(pdf_name, image_path, num_points, point_cloud_folder)[1]
    return results


def plot_point_cloud(obj_path, point_size=0.5):
    """Plot a 2D point cloud from an OBJ file."""
    points = np.loadtxt(obj_path, usecols=(1, 2))
    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], c='black', s=point_size)
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.show()


def load_obj_file(file_path):
    """Load an OBJ file and extract the 2D vertices."""
    mesh = trimesh.load(file_path)
    return mesh.vertices[:, :2]


def convert_to_tensor(vertices):
    """Convert the 2D vertices to a PyTorch tensor."""
    return torch.tensor(vertices, dtype=torch.float32)


def normalize_point_cloud(point_cloud_tensor):
    """Normalize the 2D point cloud to center it at origin and scale it."""
    point_cloud_tensor -= point_cloud_tensor.mean(dim=0)
    point_cloud_tensor /= point_cloud_tensor.abs().max()
    return point_cloud_tensor


def save_to_file(point_cloud_tensor, file_path):
    """Save the PyTorch tensor to a .pt file."""
    torch.save(point_cloud_tensor, file_path)


def load_from_file(file_path):
    """Load the PyTorch tensor from a .pt file."""
    return torch.load(file_path)


def visualize_point_cloud(point_cloud_tensor):
    """Visualize the 2D point cloud using matplotlib."""
    plt.scatter(point_cloud_tensor[:, 0], point_cloud_tensor[:, 1], s=0.0001, color='blue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Point Cloud')
    plt.axis('equal')
    plt.show()


# Example Usage
pdf_folder = r"M:\Order to PC\CAD_Reconstruction\CAD_Retrival\Image_to_PC\Pdf"
image_output_folder = r"M:\Order to PC\CAD_Reconstruction\CAD_Retrival\Image_to_PC\Image_output"
point_cloud_folder = r"M:\Order to PC\CAD_Reconstruction\CAD_Retrival\Image_to_PC\point_cloud"
num_points = 1000

# Sequential processing of PDFs and point clouds
image_paths = extract_eighth_page(pdf_folder, image_output_folder, dpi=600)
point_counts = batch_process_point_clouds(image_paths, point_cloud_folder, num_points)
