import os 
from PIL import Image
import numpy as np

# Custom distance function that handles temporal gaps intelligently
def neurite_distance(features_a, features_b, temporal_eps):
    """
    Calculate distance between two neurite observations.
    
    Args:
        features_a: [centroid_x, centroid_y, frame_index, length, dir_x, dir_y, location_flag]
        features_b: [centroid_x, centroid_y, frame_index, length, dir_x, dir_y, location_flag]
    """
    # 1. Spatial distance (Euclidean distance between centroids)
    spatial_dist = np.sqrt((features_a[0] - features_b[0])**2 + 
                        (features_a[1] - features_b[1])**2)
    
    # 2. Temporal distance with intelligent gap handling
    frame_gap = abs(features_a[2] - features_b[2])  # This is frame difference, NOT spatial!
    
    if frame_gap > temporal_eps:
        # Heavy penalty for large temporal gaps
        temporal_penalty = (frame_gap - temporal_eps) * 10  
    else:
        temporal_penalty = 0
    
    # 3. Length similarity (small weight)
    length_diff = abs(features_a[3] - features_b[3]) * 0.1
    
    # 4. Direction similarity (medium weight)
    direction_diff = (abs(features_a[4] - features_b[4]) + 
                    abs(features_a[5] - features_b[5])) * 5
    
    # 5. Location consistency (heavy penalty for different locations)
    location_penalty = 0 if features_a[6] == features_b[6] else 50
    
    # Combined distance
    total_distance = (spatial_dist + temporal_penalty + 
                    length_diff + direction_diff + location_penalty)
    
    return total_distance

def extract_number(filename):
        parts = filename.split('-')
        if len(parts) > 1:
            try:
                return int(parts[-1].split('.')[0])
            except ValueError:
                return 0
        return 0
    
def ndf_to_txt(ndf_path, txt_path):
    with open(ndf_path, 'r') as ndf_file:
        ndf_content = ndf_file.read()
        # print(ndf_content)

    with open(txt_path, 'w') as txt_file:
        txt_file.write(ndf_content)

def convert_tif_to_png(source_folder, target_folder):
    # Ensure target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Loop through all files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith(".tif"): #or filename.endswith(".tiff"):
            # Path to the current file
            file_path = os.path.join(source_folder, filename)
            # Open the TIFF image
            with Image.open(file_path) as img:
                # Convert the file name to PNG
                img = img.convert("RGB")
                # Convert the file name to PNG
                target_file = os.path.splitext(filename)[0] + ".png"
                target_path = os.path.join(target_folder, target_file)
                # Save the image as PNG
                img.save(target_path, "PNG")
                # print(f"Converted {filename} to {target_file}")

def parse_ndf_content(txt_path):
    data = {
        'Parameters': [],
        'TypeNamesColors': {},
        'ClusterNames': [],
        'Tracings': {}
    }
    current_tracing = None
    coordinates = []  # Temporary storage for coordinates within the current segment

    with open(txt_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""

        if line.startswith('//'):
            if 'Tracing' in line or 'Segment' in line or 'Parameters' in line or 'Type names and colors' in line or 'Cluster names' in line:
                # Check if we need to finalize the current segment for the current tracing
                if coordinates:
                    # Ensure there's an even number of coordinates before creating tuples
                    if len(coordinates) % 2 == 0:
                        segment_tuples = [(coordinates[j], coordinates[j+1]) for j in range(0, len(coordinates), 2)]
                        data['Tracings'].setdefault(current_tracing, []).extend(segment_tuples)
                    coordinates = []  # Reset for a new segment

                if 'Tracing' in line:
                    current_tracing = line.split()[-1]
                elif current_tracing and line.replace(' ', '').isdigit():
                    # Only add coordinates when within a tracing block
                    coordinates.extend(map(int, line.split()))
                elif 'Segment' in line or 'Parameters' in line or 'Type names and colors' in line or 'Cluster names' in line:
                    # These lines indicate a transition that may require action similar to 'Tracing'
                    continue

        elif current_tracing and line.isdigit():
            coordinates.append(int(line))

    # Finalize any remaining segment for the last tracing
    if current_tracing and coordinates:
        if len(coordinates) % 2 == 0:
            segment_tuples = [(coordinates[i], coordinates[i + 1]) for i in range(0, len(coordinates), 2)]
            data['Tracings'][current_tracing] = segment_tuples

    return data

def frames_ndf_to_txt(frames_folder, ndf_folder):
    for frame_file in os.listdir(frames_folder):
        if frame_file.endswith(".tif"):  # Adjust based on your image file type
            base_filename = os.path.splitext(frame_file)[0]
            ndf_path = os.path.join(ndf_folder, base_filename + '.ndf')
            txt_path = os.path.join(ndf_folder, base_filename + '.txt')
            
            # Convert NDF to TXT
            ndf_to_txt(ndf_path, txt_path)