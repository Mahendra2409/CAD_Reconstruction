import json
import hashlib
import os
from collections import defaultdict

def calculate_hash(filepath):
    """Calculates the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    if not os.path.exists(filepath):  # Handle missing files gracefully
        return None 
    with open(filepath, 'rb') as file:
        while True:
            chunk = file.read(4096)  # Read file in chunks
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

def find_duplicate_bodies(json_data, base_path=""):
    """
    Finds potentially duplicate bodies in the JSON data.
    Returns a dictionary where keys are tuples representing unique body characteristics,
    and values are lists of body IDs that share those characteristics.
    """
    bodies = json_data['bodies']
    potential_duplicates = defaultdict(list)

    for body_id, body_data in bodies.items():
        # --- 1. Try hashing the geometry files (most reliable) ---
        smt_path = os.path.join(base_path, body_data['smt'])
        step_path = os.path.join(base_path, body_data['step'])
        obj_path = os.path.join(base_path, body_data['obj'])
        
        smt_hash = calculate_hash(smt_path)
        step_hash = calculate_hash(step_path)
        obj_hash = calculate_hash(obj_path)

        if smt_hash and step_hash and obj_hash:
            key = (smt_hash, step_hash, obj_hash)
            potential_duplicates[key].append(body_id)
            continue  # Go to the next body if we have a hash
        
        # --- 2. Fallback: Use physical properties (less reliable) ---
        physical_props = body_data['physical_properties']
        if ('volume' not in physical_props or 
            'area' not in physical_props or 
            'xyz_moments_of_inertia' not in physical_props):
            continue

        volume = physical_props['volume']
        area = physical_props['area']
        
        # Only consider bodies with positive volume and area.
        if volume <= 0 or area <= 0:
            continue
        
        # Calculate a scale-invariant ratio (volume^(2/3) / area) rounded to reduce floating-point error.
        volume_area_ratio = round((volume**(2/3)) / area, 6)

        # Use moments of inertia (rotation-invariant).
        moments = physical_props['xyz_moments_of_inertia']
        moments_key = (
            round(moments['xx'], 6),
            round(moments['yy'], 6),
            round(moments['zz'], 6),
            round(moments.get('xy', 0), 6),
            round(moments.get('yz', 0), 6),
            round(moments.get('xz', 0), 6)
        )
        
        # Combine into a single key along with the appearance ID.
        key = (
            volume_area_ratio,
            moments_key,
            body_data['appearance']['id']
        )
        potential_duplicates[key].append(body_id)

    # Filter out groups with only one element (i.e., not duplicates).
    duplicates = {k: v for k, v in potential_duplicates.items() if len(v) > 1}
    return duplicates

def find_duplicate_occurrences(json_data):
    """
    Finds potentially duplicate occurrences based on components and the bodies involved.
    Returns a list of lists, where each inner list contains names of duplicate occurrences.
    """
    occurrences = json_data["occurrences"]
    components = json_data["components"]

    # Create dictionaries to easily access component and body names from their IDs.
    component_names = {comp_id: comp["name"] for comp_id, comp in components.items()}
    
    body_names = {}
    for comp_id, comp in components.items():
        if "bodies" in comp:
            for body_id, body in comp["bodies"].items():
                body_names[body_id] = body

    # Group occurrences by component and by the bodies involved.
    occurrences_by_component = defaultdict(lambda: defaultdict(list))
    for occ_id, occ in occurrences.items():
        component_id = occ["component"]
        if "bodies" in occ:
            bodies = tuple(occ["bodies"])
            occurrences_by_component[component_id][bodies].append(occ["name"])
        else:
            occurrences_by_component[component_id][()].append(occ["name"])

    # Filter out groups that have only one occurrence (i.e., no duplicates).
    duplicates = []
    for comp_id in occurrences_by_component:
        for bodies_key, occ_list in occurrences_by_component[comp_id].items():
            if len(occ_list) > 1:
                duplicates.append(occ_list)

    return duplicates

if __name__ == '__main__':
    # Specify the full path to your JSON file.
    json_file_path = r'M:\Order to PC\CAD_Reconstruction\CAD_Retrival\Fusion360GalleryDataset\Data\Assembly\assembly\20141_b9376856\assembly.json'  # <-- Replace with your actual JSON file path

    # Load the JSON data from the file.
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)
    
    # Use the directory containing the JSON file as the base path for resolving relative file paths.
    base_path = os.path.dirname(json_file_path)

    # Find duplicate bodies and print the results.
    duplicate_bodies = find_duplicate_bodies(json_data, base_path=base_path)
    print("Duplicate Bodies:")
    print(duplicate_bodies)

    # Find duplicate occurrences and print the results.
    duplicate_occurrences = find_duplicate_occurrences(json_data)
    print("Duplicate Occurrences:")
    print(duplicate_occurrences)
