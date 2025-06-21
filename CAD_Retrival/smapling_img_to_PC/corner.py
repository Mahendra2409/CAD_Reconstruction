import cv2
import numpy as np
import os

def detect_corners_on_solid_lines(folder_path, output_folder, min_distance=10, corner_threshold_ratio=0.005):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing: {image_path}")
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Create a solid line mask and a dashed line mask
        solid_mask = np.zeros_like(gray)
        dashed_mask = np.zeros_like(gray)
        
        # Find contours (lines) in the image
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1) #///////////////////////CHAIN_APPROX_SIMPLE
        
        # Create a color image for debugging
        debug_image = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        
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
            
            # Simple dashed line detection: Count gaps along the line
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
                if len(gaps) > 3:
                    # Check if gaps are consistent (indicating a pattern)
                    gaps = np.array(gaps)
                    mean_gap = np.mean(gaps)
                    std_gap = np.std(gaps)
                    
                    # Dashed lines have consistent gaps
                    if 2 < mean_gap < 20 and std_gap / mean_gap < 0.5:
                        is_dashed = True
                        cv2.drawContours(dashed_mask, [contour], 0, 255, 2)
                        
                        # Debug: Draw dashed lines in red
                        cv2.drawContours(debug_image, [contour], 0, (0, 0, 255), 1)
                    else:
                        cv2.drawContours(solid_mask, [contour], 0, 255, 2)
                        
                        # Debug: Draw solid lines in green
                        cv2.drawContours(debug_image, [contour], 0, (0, 255, 0), 1)
                else:
                    cv2.drawContours(solid_mask, [contour], 0, 255, 2)
                    
                    # Debug: Draw solid lines in green
                    cv2.drawContours(debug_image, [contour], 0, (0, 255, 0), 1)
            else:
                cv2.drawContours(solid_mask, [contour], 0, 255, 2)
                
                # Debug: Draw solid lines in green
                cv2.drawContours(debug_image, [contour], 0, (0, 255, 0), 1)
        
        # Save debug image
        debug_path = os.path.join(output_folder, f"debug_{image_file}")
        cv2.imwrite(debug_path, debug_image)
        print(f"Saved debug image: {debug_path}")
        
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
        
        if len(coords) == 0:
            print(f"No corners detected on solid lines in {image_path}")
            continue
        
        # Filter out closely spaced corners
        filtered_coords = []
        for point in coords:
            if all(np.linalg.norm(point - np.array(existing)) > min_distance for existing in filtered_coords):
                filtered_coords.append(point)
        
        filtered_coords = np.array(filtered_coords)
        
        if len(filtered_coords) == 0:
            print(f"All detected corners were too close and removed.")
            continue
        
        # Create output image
        result_image = cv2.imread(image_path)
        
        # Mark detected corners with red circles
        for coord in filtered_coords:
            y, x = coord
            cv2.circle(result_image, (x, y), 2, (0, 0, 255), -1)
        
        # Save result
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, result_image)
        print(f"Saved result: {output_path}")

# Example usage:
folder_path = r"M:\Order to PC\CAD_Reconstruction\CAD_Retrival\Image_to_PC\Images"
output_folder = r"M:\Order to PC\CAD_Reconstruction\CAD_Retrival\Image_to_PC\Processed_Images"
detect_corners_on_solid_lines(folder_path, output_folder)