import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from pathlib import Path
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')
from preprocessing_helpers import *
from sklearn.metrics import pairwise_distances
import traceback
from functools import partial


# Use the custom metric
class NeuritePreprocessor:
    """
    Improved neurite preprocessing pipeline with better tracking and temporal consistency.
    """
    def __init__(self, base_folder, outputs_folder="outputs", pixel_to_micron=4.9231): # pixxel per micron metadate resolution
        self.base_folder = Path(base_folder)
        self.outputs_folder = Path(outputs_folder)
        self.pixel_to_micron = pixel_to_micron
        self.outputs_folder.mkdir(exist_ok=True, parents=True)
        self.all_frame_tracings = {}  # Add this to store tracing data

    def get_folder_name(self, path):
        """Get clean folder name from path."""
        return Path(path).name
    
    def rename_files(self, frames_folder):
        files = os.listdir(frames_folder)
        ndf_files = [f for f in files if f.endswith('.ndf')]

        sorted_ndf_files = sorted(ndf_files, key=extract_number)

        for i, filename in enumerate(sorted_ndf_files, start=1):
            new_name = f"frame_{i:04d}.ndf"
            os.rename(os.path.join(frames_folder, filename), os.path.join(frames_folder, new_name))
            
            tif_filename = filename.replace('.ndf', '.tif')
            if os.path.exists(os.path.join(frames_folder, tif_filename)):
                new_tif_name = new_name.replace('.ndf', '.tif')
                os.rename(os.path.join(frames_folder, tif_filename), os.path.join(frames_folder, new_tif_name))

    def convert_tif_to_png(self, source_folder, target_folder):
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
                    target_file = os.path.splitext(filename)[0] + ".png"
                    target_path = os.path.join(target_folder, target_file)
                    # Save the image as PNG
                    img.save(target_path, "PNG")

    def frames_ndf_to_txt(self, frames_folder, ndf_folder):
        for frame_file in os.listdir(frames_folder):
            if frame_file.endswith(".tif"):  # Adjust based on your image file type
                base_filename = os.path.splitext(frame_file)[0]
                ndf_path = os.path.join(ndf_folder, base_filename + '.ndf')
                txt_path = os.path.join(ndf_folder, base_filename + '.txt')
                
                # Convert NDF to TXT
                ndf_to_txt(ndf_path, txt_path)
    
    def load_tracing_data(self, frames_folder):
        tracings_data = {}
        for frame_file in sorted(os.listdir(frames_folder)):
            if frame_file.endswith(".txt"):
                frame_index = int(frame_file.split('_')[1].split('.')[0])  
                txt_path = os.path.join(frames_folder, frame_file)
                tracings_data[frame_index] = parse_ndf_content(txt_path)
        return tracings_data

    def process_frames_and_ndf(self, frames_folder, ndf_folder, output_folder):
        """
        Process frames and NDF files. 
        """
        # Implement file conversion and data loading based on your format
        self.rename_files(frames_folder)
        self.convert_tif_to_png(frames_folder, output_folder)
        self.frames_ndf_to_txt(frames_folder, ndf_folder)
        tracing_data = self.load_tracing_data(ndf_folder)
        return tracing_data

    def load_grid_mask(self, dataset_folder):
        """
        Load the grid mask for a specific dataset.
        
        Args:
            dataset_folder (Path): Path to the dataset folder (e.g., 'mushroom_p4')
            
        Returns:
            numpy.ndarray: Binary mask where 1 indicates pillar areas, 0 indicates flat areas
        """
        mask_path = self.base_folder / dataset_folder.name / "mask" / "grid_mask.png"
        
        if not mask_path.exists():
            print(f"Warning: Grid mask not found at {mask_path}")
            return None
            
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Could not load grid mask from {mask_path}")
                return None
                
            # Normalize mask to binary (0 and 1)
            mask_binary = (mask > 0).astype(np.uint8)
            print(f"Loaded grid mask from {mask_path}, shape: {mask_binary.shape}")
            return mask_binary
            
        except Exception as e:
            print(f"Error loading grid mask from {mask_path}: {e}")
            return None

    def classify_neurites_by_location(self, frame_data, grid_mask, frame_shape=None, dataset_name=None):
        """
        Classify neurites as being on pillars (inside grid) or control (flat areas).
        Special handling for thin_p4_p10 dataset - split into thin_p4 and thin_p10.
        For all other datasets, pillar_subtype is the dataset name.
        
        Args:
            frame_data (pd.DataFrame): DataFrame containing neurite data with 'centroid_x', 'centroid_y' coordinates
            grid_mask (numpy.ndarray): Binary grid mask
            frame_shape (tuple): Shape of the original frame (height, width)
            dataset_name (str): Name of the dataset to identify p4_p10 cases
            
        Returns:
            pd.DataFrame: Updated DataFrame with 'location_type' and 'pillar_subtype' columns
        """
        if grid_mask is None:
            print("No grid mask provided, all neurites classified as 'unknown'")
            frame_data['location_type'] = 'unknown'
            frame_data['pillar_subtype'] = 'unknown'
            return frame_data
            
        if frame_data.empty:
            return frame_data
            
        # Ensure coordinates are within mask bounds
        if frame_shape is not None:
            mask_resized = cv2.resize(grid_mask, (frame_shape[1], frame_shape[0]))
        else:
            mask_resized = grid_mask
            
        # Special case: only thin_p4_p10 needs spatial splitting
        is_thin_p4_p10 = dataset_name and dataset_name.lower() == 'thin_p4_p10'
        
        # For thin_p4_p10, find the split point (approximately half, but upper part is p10)
        if is_thin_p4_p10:
            split_y = mask_resized.shape[0] // 2
            print(f"thin_p4_p10 dataset detected: split at y={split_y} (upper: thin_p10, lower: thin_p4)")
        
        # Classify each neurite based on its position
        location_types = []
        pillar_subtypes = []
        
        for idx, row in frame_data.iterrows():
            x, y = int(row['centroid_x']), int(row['centroid_y'])
            
            # Check if coordinates are within mask bounds
            if (0 <= y < mask_resized.shape[0] and 
                0 <= x < mask_resized.shape[1]):
                
                if mask_resized[y, x] == 1:
                    location_types.append('pillar')
                    
                    # Special handling for thin_p4_p10 - split into thin_p4 and thin_p10
                    if is_thin_p4_p10:
                        if y < split_y:  # Upper part - thin_p10
                            pillar_subtypes.append('thin_p10')
                        else:  # Lower part - thin_p4
                            pillar_subtypes.append('thin_p4')
                    else:
                        # For all other datasets, use the dataset name as pillar_subtype
                        pillar_subtypes.append(dataset_name if dataset_name else 'unknown')
                else:
                    location_types.append('control')
                    pillar_subtypes.append('control')
            else:
                location_types.append('out_of_bounds')
                pillar_subtypes.append('out_of_bounds')
                
        frame_data['location_type'] = location_types
        frame_data['pillar_subtype'] = pillar_subtypes
        
        # Print classification summary
        if 'pillar' in location_types:
            pillar_count = location_types.count('pillar')
            control_count = location_types.count('control')
            print(f"Location classification: pillar={pillar_count}, control={control_count}")
            
            if is_thin_p4_p10:
                p4_count = pillar_subtypes.count('thin_p4')
                p10_count = pillar_subtypes.count('thin_p10')
                print(f"Pillar subtype classification: thin_p4={p4_count}, thin_p10={p10_count}")
            else:
                # Print unique pillar subtypes
                unique_pillar_types = set([pt for pt, lt in zip(pillar_subtypes, location_types) if lt == 'pillar'])
                print(f"Pillar subtypes: {list(unique_pillar_types)}")
        
        return frame_data
    
    def prepare_tracing_features_for_tracking(self, all_frame_tracings):
        """
        Prepare tracing features for cross-frame tracking.
        Each tracing becomes a data point with its features.
        """
        all_features = []
        
        for frame_index, frame_data in all_frame_tracings.items():
            tracings = frame_data.get('Tracings', {})
            
            for tracing_id, points in tracings.items():
                features = self.calculate_neurite_features(points, tracing_id)
                if features is not None:
                    features['frame_index'] = frame_index
                    features['unique_tracing_id'] = f"f{frame_index}_t{tracing_id}"
                    # store the original points for later use
                    features['tracing_points'] = points
                    all_features.append(features)
        
        return pd.DataFrame(all_features)
    
    def calculate_path_length(self, path):
        """Calculate the length of a path given a list of coordinates."""
        if len(path) < 2:
            return 0.0
        
        length = 0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            length += math.sqrt(dx*dx + dy*dy)
        return length / self.pixel_to_micron  # Convert to microns

    def calculate_centroid(self, points):
        """Calculate the centroid from a list of points."""
        if len(points) == 0:
            return np.array([0, 0])
        return np.mean(points, axis=0)
    
    def calculate_neurite_features(self, points, tracing_id):
        """Calculate comprehensive features for a neurite tracing."""
        if len(points) < 2:
            return None
        
        points = np.array(points)
        
        # Calculate centroid
        centroid = self.calculate_centroid(points)
        
        # Calculate path length
        path_length = self.calculate_path_length(points)
        
        # Calculate orientation vector (from start to end)
        start_point = points[0]
        end_point = points[-1]
        direction_vector = end_point - start_point
        direction_magnitude = np.linalg.norm(direction_vector)
        
        # Normalize direction vector
        if direction_magnitude > 0:
            direction_vector = direction_vector / direction_magnitude
        
        # Calculate straightness and tortuosity
        if path_length > 0:
            endpoint_distance = np.linalg.norm(points[-1] - points[0]) / self.pixel_to_micron
            straightness = endpoint_distance / path_length
            tortuosity = path_length / endpoint_distance if endpoint_distance > 0 else float('inf')
        else:
            endpoint_distance = 0.0
            straightness = 0.0
            tortuosity = 0.0
        
        return {
            'tracing_id': tracing_id,
            'centroid_x': float(centroid[0]),
            'centroid_y': float(centroid[1]),
            'start_x': float(start_point[0]),
            'start_y': float(start_point[1]),
            'end_x': float(end_point[0]),
            'end_y': float(end_point[1]),
            'direction_x': float(direction_vector[0]),
            'direction_y': float(direction_vector[1]),
            'length_microns': path_length,
            'straightness': straightness,
            'tortuosity': tortuosity,
            'endpoint_distance': endpoint_distance,
            'n_points': len(points)
        }

    def create_neurite_video(self, cluster_id, cluster_data, frames_folder, output_path, movement_metrics=None, total_time_hours=24):
        """
        Create separate videos for length tracking and centroid dynamics.
        """
        # Get frame paths
        frame_files = sorted(glob.glob(str(Path(frames_folder) / "*.png")))
        if not frame_files:
            print(f"No frame files found in {frames_folder}")
            return
        
        # Get video properties from first frame
        first_frame = cv2.imread(frame_files[0])
        if first_frame is None:
            print(f"Could not read first frame: {frame_files[0]}")
            return
            
        height, width = first_frame.shape[:2]
        fps = 5  # Frames per second
        
        # Calculate time per frame in minutes
        frames = sorted(self.all_frame_tracings.keys())
        time_per_frame_minutes = total_time_hours * 60 / len(frames) if frames else 0
        
        # Colors for visualization
        colors = {
            'advancing': (0, 255, 0),    # Green
            'retracting': (0, 0, 255),   # Red
            'stable': (255, 255, 0),     # Cyan
            'text': (255, 255, 255),     # White
            'trail': (255, 165, 0),      # Orange for movement trail
            'length_indicator': (0, 255, 255),  # Yellow for length
            'centroid': (255, 0, 255),   # Magenta for centroid
            'displacement': (0, 255, 0), # Green for displacement vector
            'velocity_positive': (0, 255, 0),  # Green for positive velocity
            'velocity_negative': (0, 0, 255),  # Red for negative velocity
            'pillar': (255, 0, 0),       # Blue for pillar location
            'control': (0, 255, 0),      # Green for control location
            'edge': (255, 255, 0),       # Yellow for edge location
        }
        
        # Initialize video writers for separate videos
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Video 1: Length tracking
        length_video_path = output_path / f"cluster_{cluster_id}_length_tracking.mp4"
        length_out = cv2.VideoWriter(str(length_video_path), fourcc, fps, (width, height))
        
        # Video 2: Centroid dynamics  
        dynamics_video_path = output_path / f"cluster_{cluster_id}_centroid_dynamics.mp4"
        dynamics_out = cv2.VideoWriter(str(dynamics_video_path), fourcc, fps, (width, height))
        
        # Track data for both videos
        centroid_positions = []
        length_history = []
        frame_history = []
        location_history = []
        
        # Store previous frame data for velocity calculation
        prev_frame_data = None
        
        for frame_idx, frame_file in enumerate(frame_files, start=1):
            frame = cv2.imread(frame_file)
            if frame is None:
                continue
            
            # Initialize frames for both videos
            length_frame = frame.copy()  # For length tracking video
            dynamics_frame = frame.copy()  # For centroid dynamics video
            
            # Get current frame data
            frame_data = cluster_data[cluster_data['frame_index'] == frame_idx]
            
            # Initialize current_time_minutes with default value
            current_time_minutes = frame_idx * time_per_frame_minutes
            
            if not frame_data.empty:
                neurite_row = frame_data.iloc[0]
                current_length = neurite_row['length_microns']
                current_centroid = (int(neurite_row['centroid_x']), int(neurite_row['centroid_y']))
                current_location = neurite_row.get('location_type', 'unknown')
                current_time_minutes = neurite_row.get('time_minutes', frame_idx * time_per_frame_minutes)
                
                # Get tracing points from the stored data
                tracing_points = []
                has_tracing_column = 'tracing_points' in cluster_data.columns
                
                if has_tracing_column:
                    tracing_points = neurite_row['tracing_points']
                
                # VIDEO 1: LENGTH TRACKING
                if tracing_points and len(tracing_points) >= 2:
                    try:
                        # Create zoomed view focused on the neurite tracing
                        zoomed_frame = self.zoom_into_tracing(frame, tracing_points, zoom_padding=100)
                        print(f"Zoomed view created for frame {frame_idx} with shape {zoomed_frame.shape}")
                        # Use the zoomed frame directly for length tracking video
                        length_frame = zoomed_frame
                        
                        # Draw the neurite tracing on zoomed view
                        for i in range(1, len(tracing_points)):
                            pt1 = (int(tracing_points[i-1][0]), int(tracing_points[i-1][1]))
                            pt2 = (int(tracing_points[i][0]), int(tracing_points[i][1]))
                            cv2.line(length_frame, pt1, pt2, colors['length_indicator'], 3, cv2.LINE_AA)
                        
                        # Calculate optimal position for length annotation (on the neurite)
                        if tracing_points:
                            # Use midpoint of tracing for annotation
                            mid_idx = len(tracing_points) // 2
                            text_x = int(tracing_points[mid_idx][0])
                            text_y = int(tracing_points[mid_idx][1]) - 20
                            
                            # Ensure text stays within frame bounds
                            text_x = max(50, min(text_x, length_frame.shape[1] - 150))
                            text_y = max(30, min(text_y, length_frame.shape[0] - 10))
                            
                            # Length annotation directly on neurite
                            cv2.putText(length_frame, f"Length: {current_length:.1f} um", 
                                    (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['text'], 2)
                            
                            # Location annotation (with color coding)
                            location_color = colors.get(current_location, colors['text'])
                            cv2.putText(length_frame, f"Location: {current_location}", 
                                    (text_x, text_y + 25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, location_color, 2)
                            
                    except Exception as e:
                        print(f"Error in zoomed view for frame {frame_idx}: {e}")
                        # Keep the original frame but add error message
                        cv2.putText(length_frame, "Zoom error", 
                                (width//2-50, height//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['text'], 2)
                
                # Add length history graph to length tracking video
                length_history.append(current_length)
                frame_history.append(frame_idx)
                location_history.append(current_location)
                
                # Draw simple length history graph
                graph_width, graph_height = 250, 120
                graph_x, graph_y = 10, length_frame.shape[0] - graph_height - 10
                
                # Draw graph background
                cv2.rectangle(length_frame, (graph_x, graph_y), 
                            (graph_x + graph_width, graph_y + graph_height), (50, 50, 50), -1)
                cv2.rectangle(length_frame, (graph_x, graph_y), 
                            (graph_x + graph_width, graph_y + graph_height), colors['text'], 1)
                
                if len(length_history) > 1:
                    # Normalize values for display
                    max_length = max(length_history) if max(length_history) > 0 else 1
                    min_length = min(length_history)
                    
                    for i in range(1, len(length_history)):
                        x1 = graph_x + int((i-1) * graph_width / (len(length_history)-1))
                        y1 = graph_y + graph_height - int((length_history[i-1] - min_length) * graph_height / (max_length - min_length))
                        x2 = graph_x + int(i * graph_width / (len(length_history)-1))
                        y2 = graph_y + graph_height - int((length_history[i] - min_length) * graph_height / (max_length - min_length))
                        
                        # Color code by location if available
                        location_color = colors.get(location_history[i-1], colors['length_indicator'])
                        cv2.line(length_frame, (x1, y1), (x2, y2), location_color, 2)
                
                cv2.putText(length_frame, "Length History", (graph_x, graph_y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1)
                
                # Add frame info to length video
                frame_info = f"Frame: {frame_idx}/{len(frame_files)} | Time: {current_time_minutes:.1f} min"
                cv2.putText(length_frame, frame_info, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text'], 2)
                
                # VIDEO 2: CENTROID DYNAMICS
                # Track centroid positions
                centroid_positions.append(current_centroid)
                
                # Draw movement trail with location-based coloring
                for i in range(1, len(centroid_positions)):
                    if i-1 < len(location_history):
                        trail_color = colors.get(location_history[i-1], colors['trail'])
                    else:
                        trail_color = colors['trail']
                    cv2.line(dynamics_frame, centroid_positions[i-1], centroid_positions[i], 
                            trail_color, 3, cv2.LINE_AA)
                
                # Draw current centroid
                cv2.circle(dynamics_frame, current_centroid, 10, colors['centroid'], -1)
                cv2.circle(dynamics_frame, current_centroid, 12, colors['text'], 2)
                
                # Calculate and display dynamics metrics
                velocity = 0.0
                net_displacement = 0.0
                signed_direction = 0.0
                
                if len(centroid_positions) > 1:
                    # Calculate net displacement from first position
                    start_pos = centroid_positions[0]
                    net_dx = current_centroid[0] - start_pos[0]
                    net_dy = current_centroid[1] - start_pos[1]
                    net_displacement = np.sqrt(net_dx**2 + net_dy**2) / self.pixel_to_micron
                    signed_direction = np.arctan2(net_dy, net_dx)
                    
                    # Calculate instantaneous velocity (if we have previous frame data)
                    if prev_frame_data is not None:
                        prev_centroid = (prev_frame_data['centroid_x'], prev_frame_data['centroid_y'])
                        dx = current_centroid[0] - prev_centroid[0]
                        dy = current_centroid[1] - prev_centroid[1]
                        frame_distance = np.sqrt(dx**2 + dy**2) / self.pixel_to_micron
                        time_interval_minutes = time_per_frame_minutes
                        velocity = frame_distance / time_interval_minutes  # um/min
                        
                        # Draw ONLY instantaneous movement vector (choose one)
                        vector_scale = 5
                        end_x = int(current_centroid[0] + dx * vector_scale)
                        end_y = int(current_centroid[1] + dy * vector_scale)
                        velocity_color = colors['velocity_positive'] if velocity >= 0 else colors['velocity_negative']
                        cv2.arrowedLine(dynamics_frame, current_centroid, (end_x, end_y), 
                                    velocity_color, 3, tipLength=0.3)
                
                # Add movement metrics from calculate_movement_with_gaps if available
                cumulative_text = []
                if movement_metrics:
                    cumulative_text = [
                        "CUMULATIVE METRICS:",
                        f"Net Movement: {movement_metrics.get('net_movement', 0):+.2f} um",
                        f"Advancements: {movement_metrics.get('n_advancements', 0)}",
                        f"Retractions: {movement_metrics.get('n_retractions', 0)}",
                        f"Location Changes: {movement_metrics.get('n_location_changes', 0)}"
                    ]
                
                # Add dynamics information overlay
                dynamics_text = [
                    "INSTANTANEOUS DYNAMICS:",
                    f"Velocity: {velocity:+.2f} um/min",
                    f"Net Displacement: {net_displacement:.2f} um",
                    f"Direction: {np.degrees(signed_direction):.1f}°",
                    f"Trail Points: {len(centroid_positions)}",
                    f"Time: {current_time_minutes:.1f} min"
                ]
                
                # Combine all text
                all_text = dynamics_text + [""] + cumulative_text
                
                for i, text in enumerate(all_text):
                    y_position = 30 + i * 22
                    cv2.putText(dynamics_frame, text, (10, y_position), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1)
                
                # Create zoomed view of centroid region
                if tracing_points and len(tracing_points) >= 2:
                    try:
                        zoomed_view, adjusted_points = self.zoom_into_centroid(
                            frame, current_centroid, tracing_points, zoom_size=150
                        )
                        
                        # Overlay zoomed view on dynamics frame
                        zoom_height, zoom_width = zoomed_view.shape[:2]
                        
                        # Ensure the zoomed view fits in the corner
                        if zoom_height <= height - 20 and zoom_width <= width - 20:
                            dynamics_frame[10:10+zoom_height, width-zoom_width-10:width-10] = zoomed_view
                            
                            # Draw zoom border
                            cv2.rectangle(dynamics_frame, 
                                        (width-zoom_width-15, 5),
                                        (width-5, 5+zoom_height+10),
                                        colors['text'], 2)
                            
                            cv2.putText(dynamics_frame, "Zoomed Centroid", 
                                    (width-zoom_width-10, 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1)
                            
                    except Exception as e:
                        print(f"Error creating centroid zoom for frame {frame_idx}: {e}")
                
                # Store current frame data for next iteration
                prev_frame_data = neurite_row
            
            else:
                # No data for this frame - just show the frames without annotations
                # For length video, we can show a message
                cv2.putText(length_frame, "No neurite data", 
                        (width//2-80, height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['text'], 2)
                
                # For dynamics video, just show the frame as is
            
            # Add titles to both videos
            cv2.putText(length_frame, "LENGTH TRACKING", (width//2-100, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, colors['text'], 2)
            cv2.putText(dynamics_frame, "CENTROID DYNAMICS", (width//2-120, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, colors['text'], 2)
            
            # Add cluster info to both videos
            cluster_info = f"Cluster: {cluster_id}"
            cv2.putText(length_frame, cluster_info, (width-150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text'], 2)
            cv2.putText(dynamics_frame, cluster_info, (width-150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text'], 2)
            
            # Write frames to separate videos
            length_out.write(length_frame)
            dynamics_out.write(dynamics_frame)
        
        # Release video writers
        length_out.release()
        dynamics_out.release()
        
        print(f"Length tracking video saved: {length_video_path}")
        print(f"Centroid dynamics video saved: {dynamics_video_path}")
        
    def zoom_into_tracing(self, image, tracing_points, zoom_padding=80):
        """
        Zoom into the area around the tracing points to better visualize the neurite.
        """
        if not tracing_points:
            print("No tracing points provided for zooming.")
            return image.copy()
        
        # Convert tracing points to numpy array
        points = np.array(tracing_points, dtype=np.float32)
        
        # Calculate bounding box around tracing
        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])
        
        # Add padding
        min_x = max(0, min_x - zoom_padding)
        max_x = min(image.shape[1], max_x + zoom_padding)
        min_y = max(0, min_y - zoom_padding)
        max_y = min(image.shape[0], max_y + zoom_padding)
        
        # Calculate dimensions
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        
        # Ensure minimum size
        if bbox_width < 200:
            center_x = (min_x + max_x) / 2
            min_x = max(0, center_x - 100)
            max_x = min(image.shape[1], center_x + 100)
            bbox_width = max_x - min_x
        
        if bbox_height < 200:
            center_y = (min_y + max_y) / 2
            min_y = max(0, center_y - 100)
            max_y = min(image.shape[0], center_y + 100)
            bbox_height = max_y - min_y
        
        # Extract ROI
        zoomed_image = image[int(min_y):int(max_y), int(min_x):int(max_x)].copy()
        
        # Resize if too small for consistent display
        if zoomed_image.shape[0] < 300 or zoomed_image.shape[1] < 300:
            scale_factor = max(300/zoomed_image.shape[0], 300/zoomed_image.shape[1])
            new_width = int(zoomed_image.shape[1] * scale_factor)
            new_height = int(zoomed_image.shape[0] * scale_factor)
            zoomed_image = cv2.resize(zoomed_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        return zoomed_image
    
    def zoom_into_centroid(self, image, centroid, tracing_points, zoom_size=150, padding_color=(0, 0, 0)):
        """Enhanced zoom into centroid region with better coordinate adjustment."""
        x, y = int(centroid[0]), int(centroid[1])
        
        # Calculate required padding based on image boundaries
        pad_top = max(0, zoom_size - y)
        pad_bottom = max(0, (y + zoom_size) - image.shape[0])
        pad_left = max(0, zoom_size - x)
        pad_right = max(0, (x + zoom_size) - image.shape[1])
        
        # Apply padding
        padded_image = cv2.copyMakeBorder(
            image, 
            pad_top, pad_bottom, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, value=padding_color
        )
        
        # Adjust centroid position for padding
        padded_x = x + pad_left
        padded_y = y + pad_top
        
        # Define ROI
        start_x = padded_x - zoom_size
        end_x = padded_x + zoom_size
        start_y = padded_y - zoom_size
        end_y = padded_y + zoom_size
        
        # Extract zoomed area
        zoomed_image = padded_image[start_y:end_y, start_x:end_x].copy()
        
        # Adjust tracing points to new coordinates
        adjusted_tracing_points = []
        for point in tracing_points:
            adj_x = int(point[0] - start_x + pad_left)
            adj_y = int(point[1] - start_y + pad_top)
            adjusted_tracing_points.append((adj_x, adj_y))
        
        # Draw enhanced tracing on zoomed image
        for i in range(1, len(adjusted_tracing_points)):
            cv2.line(zoomed_image, adjusted_tracing_points[i-1], 
                    adjusted_tracing_points[i], (0, 255, 0), 3)
        
        # Draw centroid in zoomed view
        centroid_zoom_x = zoom_size
        centroid_zoom_y = zoom_size
        cv2.circle(zoomed_image, (centroid_zoom_x, centroid_zoom_y), 8, (255, 0, 255), -1)
        cv2.circle(zoomed_image, (centroid_zoom_x, centroid_zoom_y), 10, (255, 255, 255), 2)
        
        # Add crosshairs for centroid
        cv2.line(zoomed_image, (centroid_zoom_x-15, centroid_zoom_y), 
                (centroid_zoom_x+15, centroid_zoom_y), (255, 255, 255), 1)
        cv2.line(zoomed_image, (centroid_zoom_x, centroid_zoom_y-15), 
                (centroid_zoom_x, centroid_zoom_y+15), (255, 255, 255), 1)
        
        # Add scale bar (assuming 10 microns)
        scale_pixels = int(10 * self.pixel_to_micron)
        scale_y = zoom_size - 20
        cv2.line(zoomed_image, (20, scale_y), (20 + scale_pixels, scale_y), 
                (255, 255, 255), 3)
        cv2.putText(zoomed_image, "10 um", (25, scale_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return zoomed_image, adjusted_tracing_points
    
    def adaptive_eps(self, frame_index, base_eps=30, decay_factor=0.1, min_eps=10):
        """
        Adaptive epsilon for DBSCAN based on frame progression.
        Later frames may have more spread, so we adjust clustering sensitivity.
        """
        return max(min_eps, base_eps * (1 + decay_factor * frame_index))

    def cluster_neurites_across_frames(self, features_df, grid_mask, dataset_name, 
                                    spatial_eps=50, temporal_eps=5, min_samples=2,
                                    max_temporal_gap=20):
        """
        Use DBSCAN to cluster neurite appearances across ALL frames to establish identity.
        PROPERLY handles temporal gaps with adaptive epsilon.
        
        Args:
            features_df: DataFrame with neurite features from all frames
            grid_mask: Binary mask for pillar/control classification
            dataset_name: Name of the dataset for pillar subtype classification
            spatial_eps: Maximum spatial distance between neurites to be considered same cluster
            temporal_eps: Maximum temporal distance (in frames) to be considered same neurite
            min_samples: Minimum number of appearances to form a cluster
            max_temporal_gap: Maximum allowed gap between appearances for same neurite
        """
        if features_df.empty:
            return features_df
        
        # First, classify neurites by location using the grid mask
        features_df = self.classify_neurites_by_location(features_df, grid_mask, dataset_name=dataset_name)
        
        # Prepare features for DBSCAN clustering with PROPER temporal handling
        cluster_features = []
        feature_indices = []
        
        # Calculate adaptive epsilon based on frame distribution
        frame_range = features_df['frame_index'].max() - features_df['frame_index'].min()
        adaptive_spatial_eps = self.adaptive_eps(frame_range, base_eps=spatial_eps)
        
        print(f"Using adaptive spatial epsilon: {adaptive_spatial_eps:.1f} (frame range: {frame_range})")
        
        for idx, row in features_df.iterrows():
            # Combine spatial, temporal, and location features with PROPER scaling
            # Scale temporal component to be comparable to spatial distances
            temporal_component = row['frame_index'] * (adaptive_spatial_eps / temporal_eps)
            
            features = [
                row['centroid_x'],           # Spatial x
                row['centroid_y'],           # Spatial y  
                temporal_component,          # PROPERLY scaled temporal component
                row['length_microns'] * 0.1, # Length similarity (lightly weighted)
                row['direction_x'] * 20,     # Direction similarity
                row['direction_y'] * 20,     # Direction similarity
                # Add location features - encode location as numeric values
                100 if row.get('location_type') == 'pillar' else 0,  # Strong weight for pillar vs control
            ]
            cluster_features.append(features)
            feature_indices.append(idx)
        
        cluster_features = np.array(cluster_features)
        
        # Use DBSCAN with PROPER epsilon that accounts for combined feature space
        # The epsilon should be scaled to account for the multi-dimensional space
        combined_eps = np.sqrt(adaptive_spatial_eps**2 + adaptive_spatial_eps**2 + 
                            (adaptive_spatial_eps)**2)  # Account for x, y, and temporal dimensions
        
        clustering = DBSCAN(
            eps=combined_eps,  # Properly scaled for combined feature space
            min_samples=min_samples,
            metric='euclidean'
        ).fit(cluster_features)
        
        # Assign cluster labels
        features_df = features_df.copy()
        features_df['global_id'] = clustering.labels_
        
        clustered_df = features_df.iloc[feature_indices].copy()
        clustered_df['global_id'] = clustering.labels_
        
        # Print clustering results
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        n_noise = list(clustering.labels_).count(-1)
        
        print(f"DBSCAN clustering completed: {n_clusters} clusters, {n_noise} noise points")
        print(f"Cluster distribution: {np.bincount(clustering.labels_ + 1)}")  # +1 to handle -1 labels
        
        # Post-process clusters with BETTER temporal gap handling
        features_df = self._validate_temporal_consistency_improved(features_df, max_temporal_gap)
        
        
        return features_df,clustered_df

    def _validate_temporal_consistency_improved(self, df, max_temporal_gap=20):
        """
        Improved temporal consistency validation that handles gaps more intelligently.
        """
        valid_df = df.copy()
        
        for cluster_id in valid_df['global_id'].unique():
            if cluster_id == -1:  # Skip noise
                continue
                
            cluster_data = valid_df[valid_df['global_id'] == cluster_id]
            
            if len(cluster_data) > 1:
                frames = sorted(cluster_data['frame_index'])
                
                # Calculate all temporal gaps
                gaps = [frames[i+1] - frames[i] for i in range(len(frames)-1)]
                max_gap = max(gaps) if gaps else 0
                
                # Calculate temporal coverage statistics
                total_time_span = frames[-1] - frames[0] + 1
                coverage_ratio = len(frames) / total_time_span
                
                # More intelligent gap handling:
                # - Allow larger gaps if overall coverage is good
                # - Consider the number of large gaps, not just the maximum
                large_gaps = sum(1 for gap in gaps if gap > max_temporal_gap)
                
                # Location consistency
                location_types = cluster_data['location_type'].value_counts()
                dominant_location = location_types.index[0]
                location_consistency = location_types[dominant_location] / len(cluster_data)
                
                # Decision criteria for keeping/splitting cluster
                should_split = False
                
                if large_gaps > 2:  # Too many large gaps
                    should_split = True
                    reason = f"too many large gaps ({large_gaps})"
                elif max_gap > max_temporal_gap * 2:  # One extremely large gap
                    should_split = True
                    reason = f"extremely large gap ({max_gap} frames)"
                elif coverage_ratio < 0.3 and max_gap > max_temporal_gap:  # Poor coverage with large gap
                    should_split = True
                    reason = f"poor coverage ({coverage_ratio:.2f}) with large gap ({max_gap} frames)"
                elif location_consistency < 0.6:  # Very inconsistent location
                    should_split = True
                    reason = f"low location consistency ({location_consistency:.2f})"
                
                if should_split:
                    print(f"Cluster {cluster_id} split: {reason}")
                    valid_df.loc[valid_df['global_id'] == cluster_id, 'global_id'] = -1
                else:
                    # Store gap information for analysis
                    valid_df.loc[valid_df['global_id'] == cluster_id, 'max_gap'] = max_gap
                    valid_df.loc[valid_df['global_id'] == cluster_id, 'coverage_ratio'] = coverage_ratio
                    valid_df.loc[valid_df['global_id'] == cluster_id, 'large_gaps_count'] = large_gaps
        
        return valid_df

    def cluster_neurites_with_custom_metric(self, features_df, grid_mask, dataset_name,
                                       spatial_eps=50, temporal_eps=5, min_samples=2,
                                       output_folder=None):
        """
        Alternative approach: Use custom distance metric that properly handles temporal gaps.
        This is more robust for handling neurites that disappear and reappear.
        """
        if features_df.empty:
            return features_df
        
        # Classify by location first
        features_df = self.classify_neurites_by_location(features_df, grid_mask, dataset_name=dataset_name)
        
        # Prepare features with clear indexing
        features_list = []
        feature_indices = []
        
        print("Preparing features for custom distance metric...")
        for idx, row in features_df.iterrows():
            features = [
                row['centroid_x'],           # Index 0: spatial x
                row['centroid_y'],           # Index 1: spatial y  
                row['frame_index'],          # Index 2: temporal (frame number)
                row['length_microns'],       # Index 3: length
                row['direction_x'],          # Index 4: direction vector x
                row['direction_y'],          # Index 5: direction vector y
                1 if row.get('location_type') == 'pillar' else 0,  # Index 6: location flag
            ]
            features_list.append(features)
            feature_indices.append(idx)
        
        features_array = np.array(features_list)

        print("Computing pairwise distances...")    
        # Use partial to create the metric function with temporal_eps
        metric_function = partial(neurite_distance, temporal_eps=temporal_eps)        
        # Compute distance matrix
        distance_matrix = pairwise_distances(features_array, metric=metric_function)
        
        # Plot distance matrix heatmap
        if output_folder:
            self.plot_distance_matrix_heatmap(distance_matrix, features_df, output_folder, dataset_name)
        
        # Use DBSCAN with precomputed distances
        print("Performing DBSCAN clustering...")
        clustering = DBSCAN(
            eps=spatial_eps,  # This now refers to our custom distance threshold
            min_samples=min_samples,
            metric='precomputed'
        ).fit(distance_matrix)
        
        features_df = features_df.copy()
        features_df['global_id'] = clustering.labels_
        
        clustered_df = features_df.iloc[feature_indices].copy()
        clustered_df['global_id'] = clustering.labels_
        
        # Analyze cluster quality
        self.analyze_cluster_quality(features_df, distance_matrix, output_folder, dataset_name)
        
        return features_df, clustered_df

    def plot_distance_matrix_heatmap(self, distance_matrix, features_df, output_folder, dataset_name):
        """
        Plot heatmap of the distance matrix to visualize clustering structure.
        """
        print("Creating distance matrix heatmap...")
        
        # Create output directory
        heatmap_dir = Path(output_folder) / "distance_analysis"
        heatmap_dir.mkdir(exist_ok=True)
        
        # Sort by frame index for better visualization
        sort_indices = np.argsort(features_df['frame_index'].values)
        sorted_distances = distance_matrix[sort_indices][:, sort_indices]
        sorted_frames = features_df['frame_index'].values[sort_indices]
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use a diverging colormap
        im = ax.imshow(sorted_distances, cmap='viridis', aspect='auto', 
                    interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Custom Distance Metric', rotation=270, labelpad=20)
        
        # Add frame labels for some reference points
        n_ticks = min(10, len(sorted_frames))
        tick_indices = np.linspace(0, len(sorted_frames)-1, n_ticks, dtype=int)
        tick_labels = [f"F{sorted_frames[i]}" for i in tick_indices]
        
        ax.set_xticks(tick_indices)
        ax.set_yticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=45)
        ax.set_yticklabels(tick_labels)
        
        ax.set_xlabel('Frame Index (sorted)')
        ax.set_ylabel('Frame Index (sorted)')
        ax.set_title(f'Neurite Distance Matrix\n{dataset_name}\n'
                    f'({len(features_df)} observations)', fontsize=14, pad=20)
        
        # Add grid
        ax.grid(False)
        
        plt.tight_layout()
        
        # Save the plot
        heatmap_path = heatmap_dir / f"distance_matrix_{dataset_name}.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Distance matrix heatmap saved: {heatmap_path}")
        
        # Also create a smaller version showing cluster structure after clustering
        self.plot_clustered_distance_matrix(distance_matrix, features_df, heatmap_dir, dataset_name)

    def plot_clustered_distance_matrix(self, distance_matrix, features_df, output_dir, dataset_name):
        """
        Plot distance matrix sorted by cluster assignments to show clustering structure.
        """

        # Only plot if we have clusters
        if 'global_id' not in features_df.columns:
            return
        
        # Sort by cluster ID, then by frame index
        features_df_sorted = features_df.sort_values(['global_id', 'frame_index'])
        sort_indices = features_df_sorted.index
        
        # Reorder distance matrix
        clustered_distances = distance_matrix[sort_indices][:, sort_indices]
        cluster_ids = features_df_sorted['global_id'].values
        
        # Create the heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Full distance matrix
        im1 = ax1.imshow(clustered_distances, cmap='viridis', aspect='auto')
        ax1.set_title(f'Distance Matrix (Cluster Sorted)\n{dataset_name}')
        ax1.set_xlabel('Observation Index')
        ax1.set_ylabel('Observation Index')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Plot 2: Zoomed in to show cluster blocks
        # Show only first 50x50 for clarity, or all if smaller
        zoom_size = min(50, len(clustered_distances))
        im2 = ax2.imshow(clustered_distances[:zoom_size, :zoom_size], cmap='viridis', aspect='auto')
        ax2.set_title(f'Zoomed View (First {zoom_size} observations)')
        ax2.set_xlabel('Observation Index')
        ax2.set_ylabel('Observation Index')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # Add cluster boundaries to zoomed plot
        unique_clusters = np.unique(cluster_ids[:zoom_size])
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue
            cluster_mask = cluster_ids[:zoom_size] == cluster_id
            if np.any(cluster_mask):
                cluster_indices = np.where(cluster_mask)[0]
                start_idx = cluster_indices[0]
                end_idx = cluster_indices[-1]
                
                # Add rectangle around cluster
                rect = plt.Rectangle((start_idx-0.5, start_idx-0.5), 
                                end_idx - start_idx + 1, 
                                end_idx - start_idx + 1,
                                fill=False, edgecolor='red', linewidth=2)
                ax2.add_patch(rect)
        
        plt.tight_layout()
        
        cluster_heatmap_path = output_dir / f"clustered_distance_matrix_{dataset_name}.png"
        plt.savefig(cluster_heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Clustered distance matrix saved: {cluster_heatmap_path}")

    def analyze_cluster_quality(self, features_df, distance_matrix, output_folder, dataset_name):
        """
        Analyze and visualize cluster quality based on distance matrix.
        """        
        if 'global_id' not in features_df.columns:
            return
        
        valid_clusters = features_df[features_df['global_id'] != -1]
        if valid_clusters.empty:
            return
        
        # Calculate intra-cluster and inter-cluster distances
        cluster_metrics = {}
        
        for cluster_id in valid_clusters['global_id'].unique():
            cluster_indices = valid_clusters[valid_clusters['global_id'] == cluster_id].index
            cluster_mask = features_df.index.isin(cluster_indices)
            
            # Intra-cluster distances (distances within the same cluster)
            intra_cluster_distances = distance_matrix[cluster_mask][:, cluster_mask]
            intra_mean = intra_cluster_distances.mean() if len(intra_cluster_distances) > 0 else 0
            
            # Inter-cluster distances (distances to other clusters)
            other_cluster_mask = (features_df['global_id'] != cluster_id) & (features_df['global_id'] != -1)
            if np.any(other_cluster_mask):
                inter_cluster_distances = distance_matrix[cluster_mask][:, other_cluster_mask]
                inter_mean = inter_cluster_distances.mean()
            else:
                inter_mean = 0
            
            cluster_metrics[cluster_id] = {
                'size': len(cluster_indices),
                'intra_cluster_mean': intra_mean,
                'inter_cluster_mean': inter_mean,
                'separation_ratio': inter_mean / intra_mean if intra_mean > 0 else float('inf')
            }
        
        # Create cluster quality visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Cluster sizes
        cluster_sizes = [metrics['size'] for metrics in cluster_metrics.values()]
        ax1.bar(range(len(cluster_sizes)), cluster_sizes)
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Number of Observations')
        ax1.set_title('Cluster Sizes')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Intra vs Inter cluster distances
        intra_means = [metrics['intra_cluster_mean'] for metrics in cluster_metrics.values()]
        inter_means = [metrics['inter_cluster_mean'] for metrics in cluster_metrics.values()]
        
        x_pos = np.arange(len(cluster_metrics))
        width = 0.35
        ax2.bar(x_pos - width/2, intra_means, width, label='Intra-cluster', alpha=0.7)
        ax2.bar(x_pos + width/2, inter_means, width, label='Inter-cluster', alpha=0.7)
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Mean Distance')
        ax2.set_title('Intra-cluster vs Inter-cluster Distances')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Separation ratios
        separation_ratios = [metrics['separation_ratio'] for metrics in cluster_metrics.values()]
        ax3.bar(range(len(separation_ratios)), separation_ratios)
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Separation Ratio (Inter/Intra)')
        ax3.set_title('Cluster Separation Quality\n(Higher = Better Separation)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        ax4.axis('off')
        summary_text = f"Cluster Quality Summary - {dataset_name}\n\n"
        summary_text += f"Total clusters: {len(cluster_metrics)}\n"
        summary_text += f"Total observations: {len(valid_clusters)}\n"
        summary_text += f"Average cluster size: {np.mean(cluster_sizes):.1f}\n"
        summary_text += f"Average intra-cluster distance: {np.mean(intra_means):.2f}\n"
        summary_text += f"Average inter-cluster distance: {np.mean(inter_means):.2f}\n"
        summary_text += f"Average separation ratio: {np.mean(separation_ratios):.2f}\n"
        summary_text += f"Noise points: {len(features_df[features_df['global_id'] == -1])}"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        quality_path = Path(output_folder) / "distance_analysis" / f"cluster_quality_{dataset_name}.png"
        plt.savefig(quality_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Cluster quality analysis saved: {quality_path}")

    
    def calculate_movement_with_gaps(self, tracked_df):
        """
        Calculate movement metrics that account for gaps in appearance.
        FIXED: Now properly handles negative values for retraction.
        """
        movement_metrics = {}
        
        for global_id, group in tracked_df.groupby('global_id'):
            if global_id == -1:
                continue
                
            group = group.sort_values('frame_index')
            
            movements = []
            advancements = []
            retractions = []
            location_changes = []
            
            centroids = group[['centroid_x', 'centroid_y']].values
            frames = group['frame_index'].values
            lengths = group['length_microns'].values
            directions = group[['direction_x', 'direction_y']].values
            locations = group['location_type'].values if 'location_type' in group.columns else ['unknown'] * len(group)
            pillar_subtypes = group['pillar_subtype'].values if 'pillar_subtype' in group.columns else ['unknown'] * len(group)
            
            # Calculate movements between consecutive appearances
            for i in range(1, len(group)):
                frame_gap = frames[i] - frames[i-1]
                
                # Spatial movement
                dx = centroids[i][0] - centroids[i-1][0]
                dy = centroids[i][1] - centroids[i-1][1]
                movement_distance = np.sqrt(dx**2 + dy**2)
                
                # Movement direction relative to neurite orientation
                prev_direction = directions[i-1]
                movement_vector = np.array([dx, dy])
                
                # Location change tracking
                location_change = locations[i] != locations[i-1]
                pillar_subtype_change = pillar_subtypes[i] != pillar_subtypes[i-1]
                
                if location_change:
                    location_changes.append({
                        'from_frame': frames[i-1],
                        'to_frame': frames[i],
                        'from_location': locations[i-1],
                        'to_location': locations[i],
                        'from_subtype': pillar_subtypes[i-1],
                        'to_subtype': pillar_subtypes[i]
                    })
                
                if np.linalg.norm(prev_direction) > 0 and np.linalg.norm(movement_vector) > 0:
                    # Project movement onto neurite direction
                    movement_component = np.dot(movement_vector, prev_direction)
                    
                    # Normalize by frame gap to get rate
                    movement_rate = movement_component / frame_gap if frame_gap > 0 else 0
                    
                    if movement_component > 0:
                        advancements.append({
                            'distance': movement_component,  # Positive for advancement
                            'rate': movement_rate,           # Positive rate
                            'frame_gap': frame_gap,
                            'from_frame': frames[i-1],
                            'to_frame': frames[i],
                            'from_location': locations[i-1],
                            'to_location': locations[i]
                        })
                    else:
                        retractions.append({
                            'distance': movement_component,  # Negative for retraction (NOT absolute value!)
                            'rate': movement_rate,           # Negative rate
                            'frame_gap': frame_gap,
                            'from_frame': frames[i-1],
                            'to_frame': frames[i],
                            'from_location': locations[i-1],
                            'to_location': locations[i]
                        })
                
                movements.append({
                    'frame_gap': frame_gap,
                    'distance': movement_distance,
                    'dx': dx,
                    'dy': dy,
                    'location_change': location_change,
                    'subtype_change': pillar_subtype_change
                })
            
            # Calculate overall metrics - NOW WITH NEGATIVE VALUES
            total_advancement = sum(a['distance'] for a in advancements)  # Positive
            total_retraction = sum(r['distance'] for r in retractions)    # Negative
            net_movement = total_advancement + total_retraction  # Since retraction is negative
            
            # Calculate appearance statistics
            appearance_frames = len(group)
            total_frames_span = frames[-1] - frames[0] + 1
            appearance_ratio = appearance_frames / total_frames_span
            
            # Location statistics
            dominant_location = max(set(locations), key=list(locations).count)
            location_consistency = list(locations).count(dominant_location) / len(locations)
            
            movement_metrics[global_id] = {
                'total_advancement': total_advancement,
                'total_retraction': total_retraction,  # Now negative
                'net_movement': net_movement,          # Can be positive or negative
                'appearance_frames': appearance_frames,
                'total_frames_span': total_frames_span,
                'appearance_ratio': appearance_ratio,
                'n_advancements': len(advancements),
                'n_retractions': len(retractions),
                'n_location_changes': len(location_changes),
                'dominant_location': dominant_location,
                'location_consistency': location_consistency,
                'avg_advancement_rate': np.mean([a['rate'] for a in advancements]) if advancements else 0,
                'avg_retraction_rate': np.mean([r['rate'] for r in retractions]) if retractions else 0,
            }
        
        return movement_metrics

    def process_all_frames_improved(self, all_frame_tracings, frames_folder, total_time_hours=24):
        """
        Improved processing using DBSCAN for cross-frame neurite identity with location classification.
        """
        print("Preparing tracing features...")
        features_df = self.prepare_tracing_features_for_tracking(all_frame_tracings)
        
        if features_df.empty:
            print("No valid tracing features found")
            return pd.DataFrame()
        
        # Load grid mask for this dataset
        dataset_folder = Path(frames_folder).parent
        dataset_name = dataset_folder.name
        grid_mask = self.load_grid_mask(dataset_folder)
        
        print(f"Clustering {len(features_df)} neurite appearances across frames with location classification...")
        
        # Use DBSCAN to establish neurite identity across all frames with location features
        # For datasets with frequent disappearances:
        # clustered_df = self.cluster_neurites_across_frames(
        #     features_df, grid_mask, dataset_name,
        #     spatial_eps=60,           # More spatial tolerance
        #     temporal_eps=10,          # Larger temporal window  
        #     max_temporal_gap=30,      # Allow larger gaps
        #     min_samples=2             # Fewer appearances needed
        # )

        # Or use custom metric for maximum control:
        clustered_df, clustered_df_tracings = self.cluster_neurites_with_custom_metric(
            features_df, grid_mask, dataset_name,
            spatial_eps=45,
            temporal_eps=15,
            output_folder=frames_folder  # This enables the heatmaps!
        )
        
        print("Calculating movement metrics with gap handling...")
        movement_metrics = self.calculate_movement_with_gaps(clustered_df)
        metric_columns = [
            'total_advancement', 'total_retraction', 'net_movement',
            'appearance_frames', 'total_frames_span', 'appearance_ratio',
            'n_advancements', 'n_retractions', 'n_location_changes',
            'dominant_location', 'location_consistency',
            'avg_advancement_rate', 'avg_retraction_rate'
        ]
        for col in metric_columns:
            clustered_df[col] = 0.0 if col not in ['dominant_location'] else 'unknown'
        # Add movement metrics to dataframe
        for global_id, metrics in movement_metrics.items():
            mask = clustered_df['global_id'] == global_id
            for key, value in metrics.items():
                if key  in metric_columns:  # Skip nested data
                    clustered_df.loc[mask, key] = value
        
        # Add time information in hours AND minutes
        frames = sorted(all_frame_tracings.keys())
        time_per_frame_hours = total_time_hours / len(frames) if frames else 0
        time_per_frame_minutes = total_time_hours * 60 / len(frames) if frames else 0
        
        clustered_df['time_hours'] = clustered_df['frame_index'] * time_per_frame_hours
        clustered_df['time_minutes'] = clustered_df['frame_index'] * time_per_frame_minutes
        
        # Print comprehensive summary
        self._print_processing_summary(clustered_df, movement_metrics)
        
        # CREATE EXPLANATION VIDEOS FOR EACH CLUSTER
        print("\nCreating explanation videos for each cluster...")
        self.create_cluster_explanation_videos(clustered_df_tracings, movement_metrics, frames_folder, total_time_hours)
        
        return clustered_df

    def create_cluster_explanation_videos(self, clustered_df, movement_metrics, frames_folder, total_time_hours=24):
        """
        Create explanation videos for each cluster showing length changes and centroid dynamics.
        """
        # Get unique clusters (excluding noise)
        valid_clusters = clustered_df[clustered_df['global_id'] != -1]['global_id'].unique()
        print(f"Tracing in clustered_df: {clustered_df['tracing_points']} valid tracings")
        print(f"Creating videos for {len(valid_clusters)} clusters...")
        
        for cluster_id in valid_clusters:
            try:
                print(f"Creating video for cluster {cluster_id}...")
                
                # Get cluster data
                cluster_data = clustered_df[clustered_df['global_id'] == cluster_id]
                
                # Get movement metrics for this cluster
                cluster_movement = movement_metrics.get(cluster_id, {})
                
                # Create output path
                output_path = Path(frames_folder).parent / "explanation_videos"
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Create the enhanced video
                self.create_neurite_video(
                    cluster_id=cluster_id,
                    cluster_data=cluster_data,
                    frames_folder=frames_folder,
                    output_path=output_path,
                    movement_metrics=cluster_movement,  # Pass the movement metrics
                    total_time_hours=total_time_hours
                )
                
            except Exception as e:
                print(f"Error creating video for cluster {cluster_id}: {e}")
                import traceback
                traceback.print_exc()

        
    def _print_processing_summary(self, df, movement_metrics):
        """Print comprehensive processing summary with location analysis."""
        valid_neurites = df[df['global_id'] != -1]
        n_neurites = valid_neurites['global_id'].nunique()
        n_appearances = len(valid_neurites)
        
        print(f"\n=== PROCESSING SUMMARY ===")
        print(f"Unique neurites identified: {n_neurites}")
        print(f"Total appearances: {n_appearances}")
        print(f"Noise/unassigned: {len(df[df['global_id'] == -1])}")
        
        if n_neurites > 0:
            avg_appearances = n_appearances / n_neurites
            print(f"Average appearances per neurite: {avg_appearances:.1f}")
            
            # Movement summary
            total_advancement = sum(m['total_advancement'] for m in movement_metrics.values())
            total_retraction = sum(m['total_retraction'] for m in movement_metrics.values())
            print(f"Total advancement: {total_advancement:.2f} μm")
            print(f"Total retraction: {total_retraction:.2f} μm")
            print(f"Net movement: {total_advancement - total_retraction:.2f} μm")
        
        # Location-based analysis
        if 'location_type' in df.columns:
            print(f"\n=== LOCATION ANALYSIS ===")
            location_counts = df['location_type'].value_counts()
            for loc_type, count in location_counts.items():
                percentage = (count / len(df)) * 100
                print(f"{loc_type}: {count} ({percentage:.1f}%)")
            
            # Analysis of valid neurites by location
            if n_neurites > 0:
                print(f"\nValid neurites by location:")
                neurites_by_location = valid_neurites.groupby('location_type')['global_id'].nunique()
                for loc_type, count in neurites_by_location.items():
                    print(f"  {loc_type}: {count} neurites")
                
                # Pillar subtype analysis
                if 'pillar_subtype' in df.columns:
                    pillar_data = df[df['location_type'] == 'pillar']
                    if not pillar_data.empty:
                        print(f"\nPillar subtypes:")
                        subtype_counts = pillar_data['pillar_subtype'].value_counts()
                        for subtype, count in subtype_counts.items():
                            percentage = (count / len(pillar_data)) * 100
                            print(f"  {subtype}: {count} ({percentage:.1f}%)")
    
    def process_single_dataset(self, folder_path):
        """Process a single dataset folder."""
        folder_path = Path(folder_path)
        folder_name = folder_path.name
        
        print(f"\nProcessing dataset: {folder_name}")
        
        # Set up paths
        frames_folder = folder_path
        ndf_folder = folder_path
        output_folder = self.outputs_folder /  folder_name / "pngs"
        output_folder.mkdir(parents=True, exist_ok=True)

        try:
            # Process frames and NDFs
            tracing_data = self.process_frames_and_ndf(frames_folder, ndf_folder, output_folder)
            
            if not tracing_data:
                print(f"No tracing data found for {folder_name}")
                return

            # Process all frames with enhanced pipeline
            df = self.process_all_frames_improved(tracing_data, str(output_folder))
            
            if df.empty:
                print(f"No valid data generated for {folder_name}")
                return

            # Save results
            output_file = output_folder / f'clustered_data_{folder_name}.csv'
            df.to_csv(output_file, index=False)
            
            # Generate quality report
            self.generate_quality_report(df, output_folder)
            
            # Print summary
            print(f"Dataset {folder_name} processed successfully:")
            print(f"  - Total observations: {len(df)}")
            print(f"  - Global neurites: {len(df[df['global_id'] != -1]['global_id'].unique())}")
            print(f"  - Output: {output_file}")
            
            # Print cluster distribution and location statistics
            cluster_counts = df[df['global_id'] != -1]['global_id'].value_counts()
            print(f"  - Cluster size distribution: min={cluster_counts.min()}, max={cluster_counts.max()}, mean={cluster_counts.mean():.1f}")
            
            # Print location-based statistics
            if 'location_type' in df.columns:
                pillar_neurites = df[df['location_type'] == 'pillar']
                control_neurites = df[df['location_type'] == 'control']
                
                print(f"  - Pillar neurites: {len(pillar_neurites)}")
                print(f"  - Control neurites: {len(control_neurites)}")
                
                if len(pillar_neurites) > 0:
                    pillar_global_ids = pillar_neurites[pillar_neurites['global_id'] != -1]['global_id'].nunique()
                    print(f"  - Unique global neurites on pillars: {pillar_global_ids}")
                
                if len(control_neurites) > 0:
                    control_global_ids = control_neurites[control_neurites['global_id'] != -1]['global_id'].nunique()
                    print(f"  - Unique global neurites in control areas: {control_global_ids}")

        except Exception as e:
            print(f"Error processing {folder_name}: {e}")
            
            traceback.print_exc()
            
    def generate_quality_report(self, df, output_folder):
        """Generate a quality assessment report for the processed data."""
        report = []
        report.append("NEURITE PREPROCESSING QUALITY REPORT")
        report.append("=" * 50)
        report.append(f"Total frames processed: {df['frame_index'].nunique()}")
        report.append(f"Total global neurites identified: {len(df[df['global_id'] != -1]['global_id'].unique())}")
        report.append(f"Noise/unassigned observations: {len(df[df['global_id'] == -1])}")
        
        # Quality metrics per global neurite
        valid_neurites = df[df['global_id'] != -1].groupby('global_id')
        
        lifespans = []
        qualities = []
        lengths = []
        
        for gid, group in valid_neurites:
            lifespan = group['frame_index'].max() - group['frame_index'].min() + 1
            avg_length = group['length_microns'].mean()
            
            lifespans.append(lifespan)
            lengths.append(avg_length)
            
        
        if lifespans:
            report.append(f"\nNeurite Lifespan Statistics:")
            report.append(f"  Mean: {np.mean(lifespans):.1f} frames")
            report.append(f"  Median: {np.median(lifespans):.1f} frames")
            report.append(f"  Min: {min(lifespans)} frames")
            report.append(f"  Max: {max(lifespans)} frames")
            
            report.append(f"\nNeurite Quality Statistics:")
            report.append(f"  Mean quality score: {np.mean(qualities):.3f}")
            report.append(f"  Median quality score: {np.median(qualities):.3f}")
            
            report.append(f"\nNeurite Length Statistics (μm):")
            report.append(f"  Mean: {np.mean(lengths):.2f}")
            report.append(f"  Median: {np.median(lengths):.2f}")
            report.append(f"  Min: {min(lengths):.2f}")
            report.append(f"  Max: {max(lengths):.2f}")
        
        # Add location-based analysis
        if 'location_type' in df.columns:
            report_path = Path(output_folder) / 'location_analysis_report.txt'
            with open(report_path, 'w', encoding="utf-8") as f:
                f.write("Location-Based Analysis Report\n")
                f.write("=" * 40 + "\n\n")
                
                # Overall statistics
                f.write("Overall Statistics:\n")
                f.write(f"Total neurite observations: {len(df)}\n")
                
                location_counts = df['location_type'].value_counts()
                for loc_type, count in location_counts.items():
                    percentage = (count / len(df)) * 100
                    f.write(f"{loc_type.capitalize()} neurites: {count} ({percentage:.1f}%)\n")
                
                f.write("\n" + "=" * 40 + "\n\n")
                
                # Global neurite analysis by location
                f.write("Global Neurite Analysis by Location:\n")
                global_neurites = df[df['global_id'] != -1]
                if not global_neurites.empty:
                    global_stats = global_neurites.groupby('location_type')['global_id'].nunique()
                    for loc_type, count in global_stats.items():
                        f.write(f"{loc_type.capitalize()} global neurites: {count}\n")
        # Save report
        report_file = Path(output_folder) / "preprocessing_quality_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"Quality report saved to: {report_file}")

    def run_full_preprocessing(self):
        """Process all datasets in the base folder."""
        subfolders = [f for f in self.base_folder.iterdir() if f.is_dir()]
        
        print(f"Found {len(subfolders)} datasets to process")
        print(f"Input folder: {self.base_folder}")
        print(f"Output folder: {self.outputs_folder}")
        print("-" * 50)
        
        for folder in subfolders:
            self.process_single_dataset(folder)
        
        print(f"\nPreprocessing complete. Results saved to: {self.outputs_folder}")


def main():
    """Main execution function."""
    base_folder = 'data/new_movies_sorted'
    outputs_folder = 'data/outputs_by_location_vids'
    
    preprocessor = NeuritePreprocessor(base_folder, outputs_folder)
    preprocessor.run_full_preprocessing()


if __name__ == "__main__":
    main()