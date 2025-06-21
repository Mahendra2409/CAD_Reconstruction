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
        if len(pdf_document) > 6:
            page = pdf_document[7]
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


def detect_corners(image, corner_threshold_ratio=0.005, min_distance=10):
    """Detect corners on solid lines in the image."""
    # Convert to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Create a solid line mask
    solid_mask = np.zeros_like(gray)
    
    # Find contours (lines) in the image
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    
    for contour in contours:
        # Convert contour to a line mask
        line_mask = np.zeros_like(gray)
        cv2.drawContours(line_mask, [contour], 0, 255, 1)
        
        # Count number of pixels in the line
        pixel_count = np.sum(line_mask > 0)
        
        # Skip too short lines
        if pixel_count < 10:
            continue
            
        # Find all non-zero coordinates
        points = np.column_stack(np.where(line_mask > 0))
        
        # Simple dashed line detection: Check for gaps along the line
        is_dashed = False
        
        # If the points form a relatively straight line, check for gaps
        if len(points) > 10:
            # Fit a line to the points
            vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Create an array of distances along the fitted line
            dists = np.array([(p[0] - x0) * vx + (p[1] - y0) * vy for p in points])
            
            # Sort points by their distance along the line
            sorted_indices = np.argsort(dists)
            sorted_points = points[sorted_indices]
            
            # Calculate gaps between consecutive points
            gaps = []
            for i in range(1, len(sorted_points)):
                p1 = sorted_points[i-1]
                p2 = sorted_points[i]
                gap = np.linalg.norm(p2 - p1)
                if gap > 1:  # Only count significant gaps
                    gaps.append(gap)
            
            # If we have multiple gaps of similar size, it's likely a dashed line
            if len(gaps) > 2:
                # Check if gaps are consistent (indicating a pattern)
                gaps = np.array(gaps)
                mean_gap = np.mean(gaps)
                std_gap = np.std(gaps)
                
                # Dashed lines have consistent gaps
                if 2 < mean_gap < 20 and std_gap / mean_gap < 0.5:
                    is_dashed = True
                else:
                    cv2.drawContours(solid_mask, [contour], 0, 255, 2)
            else:
                cv2.drawContours(solid_mask, [contour], 0, 255, 2)
        else:
            cv2.drawContours(solid_mask, [contour], 0, 255, 2)
    
    # Apply Harris corner detection
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    
    # Threshold for corner detection
    corner_threshold = corner_threshold_ratio * corners.max()
    
    # Mask out corners that are not on solid lines
    # Dilate solid_mask to make sure we catch corners at junctions
    solid_mask_dilated = cv2.dilate(solid_mask, np.ones((1, 1), np.uint8), iterations=1)
    
    corners_on_solid = np.zeros_like(corners)
    corners_on_solid[solid_mask_dilated > 0] = corners[solid_mask_dilated > 0]
    
    # Find coordinates of corners above threshold
    coords = np.argwhere(corners_on_solid > corner_threshold)
    
    # Filter out closely spaced corners
    filtered_coords = []
    for point in coords:
        if all(np.linalg.norm(point - np.array(existing)) > min_distance for existing in filtered_coords):
            filtered_coords.append(point)
    
    return np.array(filtered_coords)

def generate_corner_points(corner_coords, image_shape, num_points, radius_range=(0, 5)):
    """
    Generate points around corners.
    
    Parameters:
    - corner_coords: Array of corner coordinates (y, x)
    - image_shape: Shape of the image (height, width)
    - num_points: Number of points to generate
    - radius_range: Range of radius for point generation (min_radius, max_radius)
    
    Returns:
    - List of points around corners
    """
    corner_points = []
    
    if len(corner_coords) == 0 or num_points == 0:
        return corner_points
    
    min_radius, max_radius = radius_range
    points_per_corner = num_points // len(corner_coords)
    extra_points = num_points % len(corner_coords)
    
    for i, corner in enumerate(corner_coords):
        y, x = corner
        current_points = points_per_corner
        if i < extra_points:
            current_points += 1
            
        for _ in range(current_points):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(min_radius, max_radius)
            point_y = int(y + r * np.sin(angle))
            point_x = int(x + r * np.cos(angle))
            
            # Ensure point is within image bounds
            if 0 <= point_y < image_shape[0] and 0 <= point_x < image_shape[1]:
                corner_points.append([point_y, point_x])
    
    return corner_points

def image_to_point_cloud(pdf_name, image_path, point_cloud_folder, total_points=2000, corner_percentage=33, corner_radius=5):
    """
    Convert image to point cloud with fixed distribution between corners and edges.
    
    Parameters:
    - pdf_name: Name of the PDF file
    - image_path: Path to the image file
    - point_cloud_folder: Folder to save the point cloud
    - total_points: Total number of points to generate (default: 2000)
    - corner_percentage: Percentage of points to allocate near corners (default: 33)
    - corner_radius: Radius around corners to distribute points (default: 5)
    """
    obj_path = os.path.join(point_cloud_folder, f"{pdf_name}.obj")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return pdf_name, 0
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create binary image
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)
    
    # Detect edges
    edges = cv2.Canny(binary, 100, 200, apertureSize=3)
    
    # Get coordinates of edge points
    edge_coords = np.column_stack(np.where(edges > 0))
    
    # Detect corners
    corner_coords = detect_corners(image)
    
    # Calculate exact number of points for each category
    corner_points_count = int(total_points * (corner_percentage / 100))
    edge_points_count = total_points - corner_points_count
    
    # Initialize array to store all points
    all_points = []
    
    # Add points from edges first
    if len(edge_coords) > edge_points_count:
        indices = np.random.choice(len(edge_coords), edge_points_count, replace=False)
        all_points.extend(edge_coords[indices])
    else:
        all_points.extend(edge_coords)
        # Update edge_points_count to actual count
        edge_points_count = len(edge_coords)
    
    # Calculate how many points we need to reach the total
    remaining_points = total_points - len(all_points)
    
    # Add corner points
    if len(corner_coords) > 0:
        # First, generate initial corner points
        initial_corner_points = generate_corner_points(
            corner_coords, 
            image.shape, 
            corner_points_count, 
            radius_range=(0, corner_radius)
        )
        all_points.extend(initial_corner_points)
        
        # If we still need more points, generate additional corner points with a larger radius
        if len(all_points) < total_points:
            additional_points_needed = total_points - len(all_points)
            additional_corner_points = generate_corner_points(
                corner_coords,
                image.shape,
                additional_points_needed,
                radius_range=(corner_radius, corner_radius * 2)
            )
            all_points.extend(additional_corner_points)
    
    # If we still need more points, use nonzero pixels from the binary image
    if len(all_points) < total_points:
        points_needed = total_points - len(all_points)
        nonzero_pixels = np.column_stack(np.where(binary > 0))
        
        # Filter out points that are already in all_points
        all_points_set = set(map(tuple, all_points))
        nonzero_pixels_filtered = [p for p in nonzero_pixels if tuple(p) not in all_points_set]
        
        if len(nonzero_pixels_filtered) > points_needed:
            indices = np.random.choice(len(nonzero_pixels_filtered), points_needed, replace=False)
            all_points.extend(np.array(nonzero_pixels_filtered)[indices])
        else:
            all_points.extend(nonzero_pixels_filtered)
    
    # Convert to numpy array for processing
    final_points = np.array(all_points)
    
    print(f"Processed {pdf_name}: {len(all_points)} points generated")
    
    # Create 3D points (with z=1)
    h = binary.shape[0]
    points = np.column_stack((final_points[:, 1], h - final_points[:, 0], np.ones(len(final_points))))
    
    # Save as OBJ file
    np.savetxt(obj_path, points, fmt='v %.1f %.1f %.1f', delimiter=' ')
    
    return pdf_name, len(final_points)


def batch_process_point_clouds(image_paths, point_cloud_folder, total_points=2000, corner_percentage=33):
    """Sequential processing of point cloud conversion with fixed distribution."""
    os.makedirs(point_cloud_folder, exist_ok=True)
    results = {}
    for pdf_name, image_path in tqdm(image_paths.items(), desc="Converting to point clouds"):
        pdf_name, point_count = image_to_point_cloud(
            pdf_name, 
            image_path, 
            point_cloud_folder, 
            total_points=total_points,
            corner_percentage=corner_percentage
        )
        results[pdf_name] = point_count
        print(f"Total points in {pdf_name}: {point_count}")
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

# Sequential processing of PDFs and point clouds
image_paths = extract_eighth_page(pdf_folder, image_output_folder, dpi=600)
point_counts = batch_process_point_clouds(
    image_paths, 
    point_cloud_folder, 
    total_points=3000,  # Fixed total points
    corner_percentage=33  # 33% near corners, 67% elsewhere
)