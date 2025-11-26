import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')
import os
import glob
from pathlib import Path
from scipy import interpolate

# Set up matplotlib for high-resolution output
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

class NeuriteAnalyzer:
    def __init__(self, file_path: str, output_dir: str = "neurite_analysis_output_combined", pixels_per_micron  = 4.9231):
        """Initialize the analyzer with data loading and preprocessing."""
        self.file_path = file_path
        self.output_dir = output_dir
        self.df = None
        self.create_output_directory()
        self.load_data()
        self.preprocess_data()
        # conversion: 4.9231 pixels per micron
        self.pixels_per_micron = pixels_per_micron
        self.microns_per_pixel = 0.203  # Clear name

    def create_output_directory(self) -> None:
        """Create output directory for saving plots."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
        
    def load_data(self) -> None:
        """Load and validate the CSV data."""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
            
    def preprocess_data(self) -> None:
        """Preprocess data: extract shape and pvalue, handle missing values."""
        # Extract shape and pvalue from pillar_subtype
        self.df[['shape', 'pvalue']] = self.df['pillar_subtype'].apply(
            lambda x: pd.Series(self._extract_shape_pvalue(x))
        )
        
        # Convert time_hours to numeric, handling any errors
        self.df['time_hours'] = pd.to_numeric(self.df['time_hours'], errors='coerce')
        
        # Create time bins for minute-level and hour-level analysis
        self.df['time_minutes'] = self.df['time_hours'] * 60
        self.df['time_hour_bin'] = np.floor(self.df['time_hours'])
        self.df['time_minute_bin'] = np.floor(self.df['time_minutes'])
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['centroid_x', 'centroid_y', 'length_microns', 'straightness', 
                          'tortuosity', 'total_advancement', 'total_retraction', 'net_movement']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                
        print("Data preprocessing completed")
        
    def _extract_shape_pvalue(self, pillar_subtype: str) -> Tuple[str, str]:
        """Extract shape and pvalue from pillar_subtype column."""
        if pd.isna(pillar_subtype) or pillar_subtype == 'control':
            return 'control', 'control'
        try:
            parts = str(pillar_subtype).split('_')
            if len(parts) >= 2:
                return parts[0], parts[1]
            else:
                return parts[0], 'unknown'
        except:
            return 'unknown', 'unknown'
    
    def plot_global_id_trajectories_with_means(self) -> None:
        """Plot trajectories of global_id clusters over time with cluster mean positions."""
        try:
            # Get global_ids with multiple time points
            time_counts = self.df.groupby('global_id')['time_hours'].nunique()
            multi_time_global_ids = time_counts[time_counts > 1].index
            
            if len(multi_time_global_ids) == 0:
                print("No global_id clusters with multiple time points found for trajectory plotting")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Calculate cluster mean centroids
            cluster_means = self.df.groupby('global_id').agg({
                'centroid_x': 'mean',
                'centroid_y': 'mean',
                'location_type': 'first',
                'shape': 'first',
                'pvalue': 'first',
            }).reset_index()
            
            # Plot 1: Individual trajectories
            colors = plt.cm.tab10(np.linspace(0, 1, len(multi_time_global_ids)))
            
            for i, global_id in enumerate(multi_time_global_ids):
                cluster_data = self.df[self.df['global_id'] == global_id].sort_values('time_hours')
                location_type = cluster_data['location_type'].iloc[0]
                shape = cluster_data['shape'].iloc[0]
                
                # Plot trajectory
                ax1.plot(cluster_data['centroid_x'], cluster_data['centroid_y'], 
                        'o-', color=colors[i], alpha=0.7, linewidth=2, markersize=4,
                        label=f'{global_id} ({location_type})')
                
                # Add time labels for some points
                if i % 5 == 0:  # Label every 5th trajectory to avoid clutter
                    for j, (_, row) in enumerate(cluster_data.iterrows()):
                        if j % 2 == 0:  # Label every other time point
                            ax1.annotate(f"{row['time_hours']:.1f}h", 
                                       (row['centroid_x'], row['centroid_y']),
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=6, alpha=0.7)
            
            ax1.set_xlabel('Centroid X')
            ax1.set_ylabel('Centroid Y')
            ax1.set_title('Neurite Trajectories Over Time\n(Colored by Global_ID)')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Cluster mean positions
            location_colors = {'control': 'blue', 'pillar': 'red'}
            shapes = cluster_means['shape'].unique()
            markers = ['o', 's', '^', 'D', 'v']  # Different markers for different shapes
            
            for i, shape_type in enumerate(shapes):
                shape_data = cluster_means[cluster_means['shape'] == shape_type]
                marker = markers[i % len(markers)]
                
                for location_type in location_colors.keys():
                    loc_data = shape_data[shape_data['location_type'] == location_type]
                    if not loc_data.empty:
                        ax2.scatter(loc_data['centroid_x'], loc_data['centroid_y'],
                                  c=location_colors[location_type], marker=marker,
                                  s=100, alpha=0.7, label=f'{shape_type} - {location_type}')
            
            ax2.set_xlabel('Centroid X')
            ax2.set_ylabel('Centroid Y')
            ax2.set_title('Cluster Mean Centroid Positions\n(Colored by Location Type)')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'cluster_trajectories_and_means_{location_type}.png'), 
                       bbox_inches='tight', dpi=300)
            plt.close()
            
            print("Saved cluster trajectories and mean positions plot")
            
        except Exception as e:
            print(f"Error plotting trajectories: {e}")
    
    def analyze_temporal_features_high_res(self) -> None:
        """Analyze features at minute-level and hour-level resolution."""
        print("\nStarting temporal feature analysis...")
        try:
            # Hour-level analysis
            self._analyze_temporal_resolution('hour')
            
            # Minute-level analysis  
            self._analyze_temporal_resolution('minute')
            
        except Exception as e:
            print(f"Error in temporal analysis: {e}")
    
    def _analyze_temporal_resolution(self, resolution: str) -> None:
        """Analyze features at specified temporal resolution."""
        if resolution == 'hour':
            time_col = 'time_hour_bin'
            time_label = 'hour'
        elif resolution == 'minute':
            time_col = 'time_minute_bin' 
            time_label = 'minute'
        else:
            raise ValueError("Resolution must be 'hour' or 'minute'")
        
        # Group by time and location type
        temporal_stats = self.df.groupby([time_col, 'location_type']).agg({
            'centroid_x': ['mean', 'std', 'count'],
            'centroid_y': ['mean', 'std', 'count'],
            'length_microns': ['mean', 'std'],
            'straightness': ['mean', 'std'],
            'tortuosity': ['mean', 'std'],
            'net_movement': ['mean', 'std'],
            'total_advancement': ['mean', 'std'],
            'total_retraction': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        temporal_stats.columns = ['_'.join(col).strip() for col in temporal_stats.columns.values]
        temporal_stats = temporal_stats.reset_index()
        
        # Save temporal statistics to CSV
        csv_filename = os.path.join(self.output_dir, f'temporal_stats_{resolution}.csv')
        temporal_stats.to_csv(csv_filename, index=False)
        print(f"Saved {resolution}-level temporal statistics to {csv_filename}")
        
        # Create comprehensive temporal plots
        self._create_temporal_plots(temporal_stats, resolution, time_label, time_col)
        
        # Calculate movement dynamics
        self._calculate_movement_dynamics(resolution)

    def _create_temporal_plots(self, temporal_stats: pd.DataFrame, resolution: str, 
                            time_label: str, time_col: str) -> None:
        """Create temporal analysis plots."""
        features = [
            ('length_microns_mean', 'Length (microns)'),
            ('straightness_mean', 'Straightness'),
            ('tortuosity_mean', 'Tortuosity'),
            ('net_movement_mean', 'Net Movement'),
            ('centroid_x_mean', 'Centroid X Position'),
            ('centroid_y_mean', 'Centroid Y Position')
        ]
        
        # Filter out features that don't exist in the data
        available_features = [(f, name) for f, name in features if f in temporal_stats.columns]
        
        if not available_features:
            print("No features available for plotting")
            return
        
        # Create subplots
        n_features = len(available_features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_features == 1:
            axes = [axes] if n_rows == 1 else np.array([axes])
        else:
            axes = axes.flatten()
        
        # Define colors and styles for location types
        location_types = temporal_stats['location_type'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(location_types)))
        location_colors = dict(zip(location_types, colors))
        location_styles = {loc: '-' for loc in location_types}  # All solid lines
        
        for idx, (feature_col, feature_name) in enumerate(available_features):
            if idx < len(axes):
                ax = axes[idx]
                
                for location_type in location_types:
                    loc_data = temporal_stats[temporal_stats['location_type'] == location_type]
                    
                    if not loc_data.empty and feature_col in loc_data.columns:
                        # Sort by time column
                        loc_data = loc_data.sort_values(time_col)
                        
                        ax.plot(loc_data[time_col], loc_data[feature_col],
                            color=location_colors[location_type],
                            linestyle=location_styles[location_type],
                            linewidth=2, marker='o', markersize=4,
                            label=f'{location_type}')
                        
                        # Add error bars if available
                        std_col = feature_col.replace('_mean', '_std')
                        if std_col in loc_data.columns:
                            ax.fill_between(loc_data[time_col],
                                        loc_data[feature_col] - loc_data[std_col],
                                        loc_data[feature_col] + loc_data[std_col],
                                        alpha=0.2, color=location_colors[location_type])
                
                ax.set_xlabel(f'{time_label.capitalize()}s')
                ax.set_ylabel(feature_name)
                ax.set_title(f'{feature_name} Over Time')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(available_features), len(axes)):
            if idx < len(axes):
                axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Fix the filename - don't use location_type variable that might not exist
        plot_filename = os.path.join(self.output_dir, f'temporal_analysis_{resolution}.png')
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved {resolution}-level temporal analysis plot")

    def _calculate_movement_dynamics(self, resolution: str) -> None:
        """Calculate movement dynamics with BOTH minute and hour velocities."""
        print(f"\nCalculating movement dynamics at {resolution}-level resolution...")
        try:
            movement_data = []
            outlier_count = 0
            
            # Biological limits (in microns)
            # MAX_DISPLACEMENT = 20  # μm between time points
            # MAX_VELOCITY_HOUR = 5  # μm/hour
            # MAX_VELOCITY_MINUTE = MAX_VELOCITY_HOUR / 60  # μm/minute
            MAX_DISPLACEMENT = 50  # μm per time step (moderate increase)
            MAX_VELOCITY_HOUR = 60  # μm/hour
            MAX_VELOCITY_MINUTE = MAX_VELOCITY_HOUR / 60  # 0.5 μm/min

            all_displacements = []
            all_velocities_hour = []
            all_velocities_min = []
            all_time_diffs = []

            for pillar_subtype in self.df['pillar_subtype'].unique():
                video_data = self.df[self.df['pillar_subtype'] == pillar_subtype]
                
                for global_id in video_data['global_id'].unique():
                    cluster_data = video_data[video_data['global_id'] == global_id].sort_values('frame_index')
                    
                    if len(cluster_data) > 1:
                        for i in range(1, len(cluster_data)):
                            prev_row = cluster_data.iloc[i-1]
                            curr_row = cluster_data.iloc[i]
                            
                            frame_diff = curr_row['frame_index'] - prev_row['frame_index']
                            time_diff_hours = curr_row['time_hours'] - prev_row['time_hours']
                            
                            # Skip large gaps and invalid time differences
                            if frame_diff > 5 or time_diff_hours <= 0:
                                outlier_count += 1
                                continue
                            
                            # Convert displacement to microns - INSTANTANEOUS between two points
                            displacement_x = (curr_row['centroid_x'] - prev_row['centroid_x']) * self.microns_per_pixel
                            displacement_y = (curr_row['centroid_y'] - prev_row['centroid_y']) * self.microns_per_pixel
                            displacement_magnitude = np.sqrt(displacement_x**2 + displacement_y**2)
                            all_displacements.append(displacement_magnitude)
                            
                            
                            # Filter by displacement
                            if displacement_magnitude > MAX_DISPLACEMENT:
                                outlier_count += 1
                                continue
                            
                            # CALCULATE BOTH VELOCITY UNITS
                            time_diff_minutes = time_diff_hours * 60
                            
                            # Velocity in microns per hour
                            velocity_hour = displacement_magnitude / time_diff_hours
                            
                            # Velocity in microns per minute  
                            velocity_minute = displacement_magnitude / time_diff_minutes
                            all_velocities_hour.append(velocity_hour)
                            all_velocities_min.append(velocity_minute)
                            all_time_diffs.append(time_diff_hours)
                            # Filter by both velocity limits
                            if (abs(velocity_hour) > MAX_VELOCITY_HOUR or 
                                abs(velocity_minute) > MAX_VELOCITY_MINUTE):
                                outlier_count += 1
                                continue
                            
                            # Calculate direction angle
                            direction_rad = np.arctan2(displacement_y, displacement_x)
                            
                            # For signed net displacement
                            if abs(direction_rad) <= np.pi/2:  # Moving mostly right/forward
                                signed_net_displacement = displacement_magnitude
                            else:  # Moving mostly left/backward
                                signed_net_displacement = -displacement_magnitude
                            
                            movement_type = 'advancement' if signed_net_displacement > 0 else 'retraction'
                            
                            # Calculate time bins for both resolutions
                            hour_start = prev_row['time_hours']
                            hour_end = curr_row['time_hours']
                            minute_start = hour_start * 60
                            minute_end = hour_end * 60
                            
                            movement_data.append({
                                'global_id': global_id,
                                'pillar_subtype': pillar_subtype,
                                'location_type': curr_row['location_type'],
                                'shape': curr_row['shape'],
                                'pvalue': curr_row['pvalue'],
                                'location_type_detailed': f"{curr_row['shape']}_{curr_row['pvalue']}",
                                
                                # FRAME AND TIME INFORMATION
                                'frame_start': prev_row['frame_index'],
                                'frame_end': curr_row['frame_index'],
                                'frame_diff': frame_diff,
                                
                                # RAW TIME
                                'time_hours_start': prev_row['time_hours'],
                                'time_hours_end': curr_row['time_hours'],
                                'time_diff_hours': time_diff_hours,
                                'time_diff_minutes': time_diff_minutes,
                                
                                # TIME BINS FOR PLOTTING (both resolutions)
                                'hour_start': hour_start,
                                'hour_end': hour_end,
                                'minute_start': minute_start,
                                'minute_end': minute_end,
                                
                                # Displacement in microns
                                'displacement_x': displacement_x,
                                'displacement_y': displacement_y,
                                'displacement_magnitude': displacement_magnitude,
                                'net_displacement': signed_net_displacement,
                                
                                # VELOCITY IN BOTH UNITS
                                'signed_velocity_um_per_hour': velocity_hour,
                                'signed_velocity_um_per_minute': velocity_minute,
                                'magnitude_velocity_um_per_hour': abs(velocity_hour),
                                'magnitude_velocity_um_per_minute': abs(velocity_minute),
                                
                                'movement_type': movement_type,
                                'length_change': curr_row['length_microns'] - prev_row['length_microns'],
                                
                                # Direction info
                                'signed_direction_rad': direction_rad,
                                'signed_direction_deg': np.degrees(direction_rad),
                            })
            
            print(f"\n🔍 MOVEMENT ANALYSIS SUMMARY:")
            print(f"Pixel resolution: {self.pixels_per_micron:.4f} pixels per micron")
            print(f"Conversion factor: {self.microns_per_pixel:.4f} microns per pixel")
            print(f"Realistic movements: {len(movement_data)}")
            print(f"Filtered outliers: {outlier_count}")
            
            if all_displacements:
                print(f"Raw displacement stats: min={np.min(all_displacements):.2f}, "
                    f"median={np.median(all_displacements):.2f}, "
                    f"95th={np.percentile(all_displacements,95):.2f}, "
                    f"max={np.max(all_displacements):.2f}")
                print(f"Raw velocity stats (min): min={np.min(all_velocities_min):.3f}, "
                    f"median={np.median(all_velocities_min):.3f}, "
                    f"95th={np.percentile(all_velocities_min,95):.3f}, "
                    f"max={np.max(all_velocities_min):.3f}")
                print(f"Frame time differences (hours): median={np.median(all_time_diffs):.3f}, "
                    f"unique={np.unique(np.round(all_time_diffs,3))}")
            
            if movement_data:
                movement_df = pd.DataFrame(movement_data)
                if len(movement_df) > 0:
                    print(f"Displacement range: {movement_df['displacement_magnitude'].min():.2f} to {movement_df['displacement_magnitude'].max():.2f} μm")
                    print(f"Velocity (hour): {movement_df['signed_velocity_um_per_hour'].min():.4f} to {movement_df['signed_velocity_um_per_hour'].max():.4f} μm/hour")
                    print(f"Velocity (minute): {movement_df['signed_velocity_um_per_minute'].min():.4f} to {movement_df['signed_velocity_um_per_minute'].max():.4f} μm/minute")
                    print(f"Movement types: {movement_df['movement_type'].value_counts().to_dict()}")
                    
                    # Generate plots for both resolutions
                    for res in ['hour', 'minute']:
                        print(f"\nGenerating {res}-level plots...")
                        self._plot_polar_directions(movement_df, res)
                        self._plot_polar_directions_straight_lines(movement_df, res)
                        self._plot_movement_dynamics(movement_df, res, res)
                    # Save results
                    filename = os.path.join(self.output_dir, 'movement_dynamics_complete.csv')
                    movement_df.to_csv(filename, index=False)
                    print(f"Saved complete movement dynamics to {filename}")
                    
                    return movement_df
                else:
                    print("No movement data to save")
            else:
                print("No realistic movement data found")
                return None
                    
        except Exception as e:
            print(f"Error in movement analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def _plot_polar_directions(self, movement_df: pd.DataFrame, resolution: str) -> None:
        """Plot polar plots with signed directions preserving negative angles."""
        try:
            # Identify control group (assuming it's labeled as 'control' in location_type)
            control_mask = movement_df['location_type'] == 'control'
            experimental_data = movement_df[~control_mask]
            
            if not control_mask.any():
                print("No control group found for polar plot comparison")
                return
            
            # Get unique shapes and order by pvalue
            shapes = experimental_data['shape'].unique()
            
            # Create ordered groups by shape and pvalue
            ordered_groups = []
            for shape in shapes:
                # Get pvalues for this shape and sort them (p4, p10, p30)
                shape_data = experimental_data[experimental_data['shape'] == shape]
                pvalues = shape_data['location_type_detailed'].str.extract(r'_p(\d+)')[0].astype(int).unique()
                pvalues.sort()
                
                for pval in pvalues:
                    group_name = f"{shape}_p{pval}"
                    if group_name in experimental_data['location_type_detailed'].values:
                        ordered_groups.append(group_name)
            
            # Add control at the beginning
            all_groups = ['control'] + ordered_groups
            
            # Create figure with appropriate number of subplots
            n_groups = len(all_groups)
            n_cols = min(3, n_groups)
            n_rows = (n_groups + n_cols - 1) // n_cols
            # Define fixed radial limits for all subplots
            FIXED_MAX_RADIUS = 27.0
            RADIAL_TICKS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]  # Consistent tick marks
            fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
            
            # Create polar plots for each group
            for i, group in enumerate(all_groups, 1):
                if i > n_rows * n_cols:  # Safety check
                    break
                        
                ax = fig.add_subplot(n_rows, n_cols, i, projection='polar')
                
                if group == 'control':
                    group_data = movement_df[control_mask]
                    group_label = 'Control'
                else:
                    group_data = movement_df[movement_df['location_type_detailed'] == group]
                    group_label = group
                
                if not group_data.empty:
                    # USE SIGNED DIRECTIONS DIRECTLY - no conversion to 0-360!
                    theta_data = group_data['signed_direction_rad'].values
                    magnitude_data = group_data['displacement_magnitude'].values
                    
                    # Set fixed radial limits for ALL plots
                    ax.set_ylim(0, FIXED_MAX_RADIUS)
                    ax.set_yticks(RADIAL_TICKS)
                    
                    # Create finer bins for smoother interpolation
                    n_bins = 72  # Double the bins for smoother curves
                    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
                    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    
                    # Calculate mean and bootstrapped confidence intervals per direction bin
                    mean_magnitudes = []
                    lower_ci = []
                    upper_ci = []
                    
                    for j in range(len(bin_edges) - 1):
                        bin_mask = (theta_data >= bin_edges[j]) & (theta_data < bin_edges[j + 1])
                        if j == len(bin_edges) - 2:  # Last bin includes the upper edge
                            bin_mask = bin_mask | (theta_data == bin_edges[j + 1])
                        
                        bin_magnitudes = magnitude_data[bin_mask]
                        
                        if len(bin_magnitudes) > 0:
                            mean_magnitudes.append(np.mean(bin_magnitudes))
                            
                            # Bootstrap confidence interval
                            if len(bin_magnitudes) >= 5:  # Only bootstrap if we have enough data
                                n_bootstrap = 1000
                                bootstrap_means = []
                                for _ in range(n_bootstrap):
                                    # Sample with replacement
                                    bootstrap_sample = np.random.choice(bin_magnitudes, size=len(bin_magnitudes), replace=True)
                                    bootstrap_means.append(np.mean(bootstrap_sample))
                                
                                lower_ci.append(np.percentile(bootstrap_means, 16))  # 1 sigma (~68% CI)
                                upper_ci.append(np.percentile(bootstrap_means, 84))
                            else:
                                # For small samples, use std or a small fraction of mean
                                std_val = np.std(bin_magnitudes) if len(bin_magnitudes) > 1 else mean_magnitudes[-1] * 0.1
                                lower_ci.append(mean_magnitudes[-1] - std_val)
                                upper_ci.append(mean_magnitudes[-1] + std_val)
                        else:
                            # No data in this bin
                            mean_magnitudes.append(0)
                            lower_ci.append(0)
                            upper_ci.append(0)
                    
                    mean_magnitudes = np.array(mean_magnitudes)
                    lower_ci = np.array(lower_ci)
                    upper_ci = np.array(upper_ci)
                    
                    # Apply smoothing to fill gaps and create continuous curves                    
                    # Identify bins with data
                    valid_mask = mean_magnitudes > 0
                    if np.sum(valid_mask) >= 3:  # Need at least 3 points for interpolation
                        valid_centers = bin_centers[valid_mask]
                        valid_means = mean_magnitudes[valid_mask]
                        valid_lower = lower_ci[valid_mask]
                        valid_upper = upper_ci[valid_mask]
                        
                        # Create interpolation functions (wrap around for circular data)
                        valid_centers_wrap = np.concatenate([valid_centers - 2*np.pi, valid_centers, valid_centers + 2*np.pi])
                        valid_means_wrap = np.concatenate([valid_means, valid_means, valid_means])
                        valid_lower_wrap = np.concatenate([valid_lower, valid_lower, valid_lower])
                        valid_upper_wrap = np.concatenate([valid_upper, valid_upper, valid_upper])
                        
                        # Cubic spline interpolation for smooth curves
                        try:
                            mean_interp = interpolate.CubicSpline(valid_centers_wrap, valid_means_wrap)
                            lower_interp = interpolate.CubicSpline(valid_centers_wrap, valid_lower_wrap)
                            upper_interp = interpolate.CubicSpline(valid_centers_wrap, valid_upper_wrap)
                            
                            # Create fine grid for smooth plotting
                            theta_fine = np.linspace(-np.pi, np.pi, 360)
                            mean_fine = mean_interp(theta_fine)
                            lower_fine = lower_interp(theta_fine)
                            upper_fine = upper_interp(theta_fine)
                            
                            # Ensure non-negative values
                            mean_fine = np.maximum(mean_fine, 0)
                            lower_fine = np.maximum(lower_fine, 0)
                            upper_fine = np.maximum(upper_fine, 0)
                            
                        except:
                            # Fallback to linear interpolation if cubic fails
                            mean_interp = interpolate.interp1d(valid_centers_wrap, valid_means_wrap, kind='linear', fill_value='extrapolate')
                            lower_interp = interpolate.interp1d(valid_centers_wrap, valid_lower_wrap, kind='linear', fill_value='extrapolate')
                            upper_interp = interpolate.interp1d(valid_centers_wrap, valid_upper_wrap, kind='linear', fill_value='extrapolate')
                            
                            theta_fine = np.linspace(-np.pi, np.pi, 360)
                            mean_fine = mean_interp(theta_fine)
                            lower_fine = lower_interp(theta_fine)
                            upper_fine = upper_interp(theta_fine)
                    else:
                        # Not enough data for interpolation, use raw binned data
                        theta_fine = np.concatenate([bin_centers, [bin_centers[0]]])
                        mean_fine = np.concatenate([mean_magnitudes, [mean_magnitudes[0]]])
                        lower_fine = np.concatenate([lower_ci, [lower_ci[0]]])
                        upper_fine = np.concatenate([upper_ci, [upper_ci[0]]])
                    
                    # Plot shaded confidence interval
                    ax.fill_between(theta_fine, lower_fine, upper_fine, 
                                alpha=0.3, color='blue', label='68% CI')
                    
                    # Plot mean line
                    ax.plot(theta_fine, mean_fine, 'b-', linewidth=2, label='Mean Magnitude')
                    
                    # Configure the polar plot to show negative angles
                    ax.set_theta_offset(np.pi/2)  # Set 0° to the top
                    ax.set_theta_direction(-1)    # Clockwise direction
                    
                    # Set theta limits to show negative angles
                    ax.set_thetamin(-180)
                    ax.set_thetamax(180)
                    
                    # Customize grid and labels for signed plot
                    ax.set_xticks(np.arange(-180, 181, 45) * np.pi/180)
                    ax.set_xticklabels(['-180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°', '180°'])
                    
                    # Add quadrant lines and labels
                    max_radius = max(upper_fine) if len(upper_fine) > 0 else 1
                    quadrant_angles = [-np.pi, -np.pi/2, 0, np.pi/2]
                    quadrant_labels = ['Left (-180°)', 'Down (-90°)', 'Right (0°)', 'Up (90°)']
                    quadrant_colors = ['red', 'purple', 'green', 'blue']
                    
                    for angle, label, color in zip(quadrant_angles, quadrant_labels, quadrant_colors):
                        ax.plot([angle, angle], [0, FIXED_MAX_RADIUS * 0.8], 
                            color=color, linestyle='--', alpha=0.5, linewidth=1)
                        # Place label inside the plot
                        label_radius = FIXED_MAX_RADIUS * 0.9
                        ax.text(angle, label_radius, label, 
                            ha='center', va='center', 
                            fontsize=8, color=color, 
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                    # Add Cartesian axis lines
                    ax.plot([-np.pi, np.pi], [0, 0], 'k-', alpha=0.3, linewidth=0.5)  # Horizontal axis
                    ax.plot([-np.pi/2, np.pi/2], [0, 0], 'k-', alpha=0.3, linewidth=0.5)  # Vertical axis
                    
                    ax.set_title(f'{group_label}\n(n={len(group_data)} movements)', pad=20, fontsize=12)
                    ax.grid(True, alpha=0.3)
                    
                    # Set radial limits
                    # if max_radius > 0:
                    #     ax.set_ylim(0, max_radius * 1.1)
                    
                    # Add legend for the first subplot only to save space
                    if i == 1:
                        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                    else:
                        # Add data quality indicator
                        n_movements = len(group_data)
                        coverage = np.sum(valid_mask) / len(valid_mask) * 100
                        ax.text(0.5, -0.15, f'Data coverage: {coverage:.1f}%', 
                            transform=ax.transAxes, ha='center', fontsize=8, 
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                else:
                    ax.set_ylim(0, FIXED_MAX_RADIUS)
                    ax.set_yticks(RADIAL_TICKS)
                    ax.set_title(f'{group_label}\n(No data)', pad=20, fontsize=12)
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                        ha='center', va='center', fontsize=12, color='red')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'polar_directions_{resolution}.png'),
                    bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Saved polar direction plots for {resolution}")
            
        except Exception as e:
            print(f"Error creating polar direction plots: {e}")
            import traceback
            traceback.print_exc()

    def _plot_polar_directions_straight_lines(self, movement_df: pd.DataFrame, resolution: str) -> None:
        """Plot polar plots with signed directions using straight lines (no spline interpolation)."""
        try:
            # Identify control group (assuming it's labeled as 'control' in location_type)
            control_mask = movement_df['location_type'] == 'control'
            experimental_data = movement_df[~control_mask]
            
            if not control_mask.any():
                print("No control group found for polar plot comparison")
                return
            
            # Get unique shapes and order by pvalue
            shapes = experimental_data['shape'].unique()
            
            # Create ordered groups by shape and pvalue
            ordered_groups = []
            for shape in shapes:
                # Get pvalues for this shape and sort them (p4, p10, p30)
                shape_data = experimental_data[experimental_data['shape'] == shape]
                pvalues = shape_data['location_type_detailed'].str.extract(r'_p(\d+)')[0].astype(int).unique()
                pvalues.sort()
                
                for pval in pvalues:
                    group_name = f"{shape}_p{pval}"
                    if group_name in experimental_data['location_type_detailed'].values:
                        ordered_groups.append(group_name)
            
            # Add control at the beginning
            all_groups = ['control'] + ordered_groups
            
            # Create figure with appropriate number of subplots
            n_groups = len(all_groups)
            n_cols = min(3, n_groups)
            n_rows = (n_groups + n_cols - 1) // n_cols
            
            fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
            
            # Create polar plots for each group
            for i, group in enumerate(all_groups, 1):
                if i > n_rows * n_cols:  # Safety check
                    break
                        
                ax = fig.add_subplot(n_rows, n_cols, i, projection='polar')
                
                if group == 'control':
                    group_data = movement_df[control_mask]
                    group_label = 'Control'
                else:
                    group_data = movement_df[movement_df['location_type_detailed'] == group]
                    group_label = group
                
                if not group_data.empty:
                    # USE SIGNED DIRECTIONS DIRECTLY - no conversion to 0-360!
                    theta_data = group_data['signed_direction_rad'].values
                    magnitude_data = group_data['displacement_magnitude'].values
                    
                    # Create bins for direction
                    n_bins = 36  # Reasonable number of bins for straight lines
                    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
                    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    
                    # Calculate mean and bootstrapped confidence intervals per direction bin
                    mean_magnitudes = []
                    lower_ci = []
                    upper_ci = []
                    
                    for j in range(len(bin_edges) - 1):
                        bin_mask = (theta_data >= bin_edges[j]) & (theta_data < bin_edges[j + 1])
                        if j == len(bin_edges) - 2:  # Last bin includes the upper edge
                            bin_mask = bin_mask | (theta_data == bin_edges[j + 1])
                        
                        bin_magnitudes = magnitude_data[bin_mask]
                        
                        if len(bin_magnitudes) > 0:
                            mean_magnitudes.append(np.mean(bin_magnitudes))
                            
                            # Bootstrap confidence interval
                            if len(bin_magnitudes) >= 5:  # Only bootstrap if we have enough data
                                n_bootstrap = 1000
                                bootstrap_means = []
                                for _ in range(n_bootstrap):
                                    # Sample with replacement
                                    bootstrap_sample = np.random.choice(bin_magnitudes, size=len(bin_magnitudes), replace=True)
                                    bootstrap_means.append(np.mean(bootstrap_sample))
                                
                                lower_ci.append(np.percentile(bootstrap_means, 16))  # 1 sigma (~68% CI)
                                upper_ci.append(np.percentile(bootstrap_means, 84))
                            else:
                                # For small samples, use std or a small fraction of mean
                                std_val = np.std(bin_magnitudes) if len(bin_magnitudes) > 1 else mean_magnitudes[-1] * 0.1
                                lower_ci.append(mean_magnitudes[-1] - std_val)
                                upper_ci.append(mean_magnitudes[-1] + std_val)
                        else:
                            # No data in this bin
                            mean_magnitudes.append(0)
                            lower_ci.append(0)
                            upper_ci.append(0)
                    
                    mean_magnitudes = np.array(mean_magnitudes)
                    lower_ci = np.array(lower_ci)
                    upper_ci = np.array(upper_ci)
                    
                    # STRAIGHT LINES APPROACH - no interpolation
                    # Create closed polygon by wrapping around
                    theta_plot = np.concatenate([bin_centers, [bin_centers[0]]])
                    mean_plot = np.concatenate([mean_magnitudes, [mean_magnitudes[0]]])
                    lower_plot = np.concatenate([lower_ci, [lower_ci[0]]])
                    upper_plot = np.concatenate([upper_ci, [upper_ci[0]]])
                    
                    # Plot shaded confidence interval as straight lines
                    ax.fill_between(theta_plot, lower_plot, upper_plot, 
                                alpha=0.3, color='blue')#,  
                                #label='68% CI', step='mid')  # Use step for straight lines between bins
                    
                    # Plot mean line as straight segments
                    # ax.plot(theta_plot, mean_plot, 'b-', linewidth=2, label='Mean Magnitude', 
                    #     drawstyle='steps-mid')  # Straight lines between points
                    
                    # Alternative: Plot as regular line (still straight segments between actual data points)
                    ax.plot(theta_plot, mean_plot, 'b-', linewidth=2, label='Mean Magnitude')
                    
                    # Configure the polar plot to show negative angles
                    ax.set_theta_offset(np.pi/2)  # Set 0° to the top
                    ax.set_theta_direction(-1)    # Clockwise direction
                    
                    # Set theta limits to show negative angles
                    ax.set_thetamin(-180)
                    ax.set_thetamax(180)
                    
                    # Customize grid and labels for signed plot
                    ax.set_xticks(np.arange(-180, 181, 45) * np.pi/180)
                    ax.set_xticklabels(['-180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°', '180°'])
                    
                    # Add quadrant lines and labels
                    max_radius = max(upper_plot) if len(upper_plot) > 0 else 1
                    quadrant_angles = [-np.pi, -np.pi/2, 0, np.pi/2]
                    quadrant_labels = ['Left (-180°)', 'Down (-90°)', 'Right (0°)', 'Up (90°)']
                    quadrant_colors = ['red', 'purple', 'green', 'blue']
                    
                    for angle, label, color in zip(quadrant_angles, quadrant_labels, quadrant_colors):
                        ax.plot([angle, angle], [0, max_radius * 0.8], 
                            color=color, linestyle='--', alpha=0.5, linewidth=1)
                        # Place label inside the plot
                        label_radius = max_radius * 0.9
                        ax.text(angle, label_radius, label, 
                            ha='center', va='center', 
                            fontsize=8, color=color, 
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                    
                    # Add Cartesian axis lines
                    ax.plot([-np.pi, np.pi], [0, 0], 'k-', alpha=0.3, linewidth=0.5)  # Horizontal axis
                    ax.plot([-np.pi/2, np.pi/2], [0, 0], 'k-', alpha=0.3, linewidth=0.5)  # Vertical axis
                    
                    ax.set_title(f'{group_label}\n(n={len(group_data)} movements)', pad=20, fontsize=12)
                    ax.grid(True, alpha=0.3)
                    
                    # Set radial limits
                    if max_radius > 0:
                        ax.set_ylim(0, max_radius * 1.1)
                    
                    # Add legend for the first subplot only to save space
                    if i == 1:
                        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                    else:
                        # Add data quality indicator
                        n_movements = len(group_data)
                        coverage = np.sum(mean_magnitudes > 0) / len(mean_magnitudes) * 100
                        ax.text(0.5, -0.15, f'Data coverage: {coverage:.1f}%', 
                            transform=ax.transAxes, ha='center', fontsize=8, 
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                else:
                    ax.set_title(f'{group_label}\n(No data)', pad=20, fontsize=12)
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                        ha='center', va='center', fontsize=12, color='red')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'polar_directions_straight_lines_{resolution}.png'),
                    bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Saved polar direction plots with straight lines for {resolution}")
            
        except Exception as e:
            print(f"Error creating polar direction plots with straight lines: {e}")
            import traceback
            traceback.print_exc()
                
    def _plot_movement_dynamics(self, movement_df: pd.DataFrame, resolution: str, time_label: str) -> None:
        """Plot movement dynamics analysis with better temporal progression."""
        try:
            # Create two separate figure sets: one for temporal progression, one for relationships
            
            if resolution == 'minute':
                velocity_col = 'signed_velocity_um_per_minute'
                velocity_label = 'Velocity (μm/minute)'
                title_suffix = 'per Minute'
            else:  # Default to hour
                velocity_col = 'signed_velocity_um_per_hour' 
                velocity_label = 'Velocity (μm/hour)'
                title_suffix = 'per Hour'
            
            # Check if velocity column exists
            if velocity_col not in movement_df.columns:
                print(f"Velocity column {velocity_col} not found, using hour data")
                velocity_col = 'signed_velocity_um_per_hour'
                velocity_label = 'Velocity (μm/hour)'
                title_suffix = 'per Hour'
            
            # FIGURE 1: TEMPORAL PROGRESSION PLOTS
            fig1, axes1 = plt.subplots(3, 2, figsize=(20, 15))
            axes1 = axes1.flatten()
            
            time_col = f'{time_label.lower()}_start'
            
            # Plot 1: Signed velocity over time with trend lines
            for location_type in movement_df['location_type_detailed'].unique():
                loc_data = movement_df[movement_df['location_type_detailed'] == location_type].sort_values(time_col)
                if not loc_data.empty:
                    # Raw scatter points
                    axes1[0].scatter(loc_data[time_col], loc_data[velocity_col],
                                alpha=0.6, label=location_type, s=30)
                    # Rolling average trend line
                    if len(loc_data) > 5:
                        rolling_avg = loc_data[velocity_col].rolling(window=5, center=True).mean()
                        axes1[0].plot(loc_data[time_col], rolling_avg, 
                                    linewidth=2, alpha=0.8)
            
            axes1[0].set_xlabel(f'Time ({time_label}s)')
            axes1[0].set_ylabel('Signed Velocity')
            axes1[0].set_title('Velocity Over Time\n(Points + Rolling Average)')
            axes1[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes1[0].grid(True, alpha=0.3)
            axes1[0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero Velocity')
            
            # Plot 2: Length change over time with trend lines
            for location_type in movement_df['location_type_detailed'].unique():
                loc_data = movement_df[movement_df['location_type_detailed'] == location_type].sort_values(time_col)
                if not loc_data.empty:
                    axes1[1].scatter(loc_data[time_col], loc_data['length_change'],
                                alpha=0.6, label=location_type, s=30)
                    if len(loc_data) > 5:
                        rolling_avg = loc_data['length_change'].rolling(window=5, center=True).mean()
                        axes1[1].plot(loc_data[time_col], rolling_avg, 
                                    linewidth=2, alpha=0.8)
            
            axes1[1].set_xlabel(f'Time ({time_label}s)')
            axes1[1].set_ylabel('Length Change (microns)')
            axes1[1].set_title('Length Change Over Time\n(Points + Rolling Average)')
            axes1[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes1[1].grid(True, alpha=0.3)
            axes1[1].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No Change')
            
            # Plot 3: Cumulative displacement over time
            for location_type in movement_df['location_type_detailed'].unique():
                loc_data = movement_df[movement_df['location_type_detailed'] == location_type].sort_values(time_col)
                if not loc_data.empty:
                    cumulative_displacement = loc_data['net_displacement'].cumsum()
                    axes1[2].plot(loc_data[time_col], cumulative_displacement,
                                linewidth=2, label=location_type, marker='o', markersize=3)
            
            axes1[2].set_xlabel(f'Time ({time_label}s)')
            axes1[2].set_ylabel('Cumulative Displacement')
            axes1[2].set_title('Cumulative Displacement Over Time\n(Net Position Change)')
            axes1[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes1[2].grid(True, alpha=0.3)
            axes1[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # Plot 4: Movement type transitions over time
            for location_type in movement_df['location_type_detailed'].unique():
                loc_data = movement_df[movement_df['location_type_detailed'] == location_type].sort_values(time_col)
                if not loc_data.empty:
                    # Encode movement types: advancement=1, retraction=-1
                    movement_encoded = loc_data['movement_type'].map({'advancement': 1, 'retraction': -1})
                    axes1[3].scatter(loc_data[time_col], movement_encoded,
                                alpha=0.7, label=location_type, s=50)
            
            axes1[3].set_xlabel(f'Time ({time_label}s)')
            axes1[3].set_ylabel('Movement Type')
            axes1[3].set_title('Movement Type Over Time\n(1=Advancement, -1=Retraction)')
            axes1[3].set_yticks([-1, 1])
            axes1[3].set_yticklabels(['Retraction', 'Advancement'])
            axes1[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes1[3].grid(True, alpha=0.3)
            
            # Plot 5: Velocity distribution by time bins (boxplot over time)
            time_bins = pd.cut(movement_df[time_col], bins=10)
            sns.boxplot(data=movement_df, x=time_bins, y=velocity_col, 
                    ax=axes1[4], palette='viridis')
            axes1[4].set_title('Velocity Distribution Across Time Bins')
            axes1[4].set_ylabel('Signed Velocity')
            axes1[4].set_xlabel(f'Time ({time_label}s) Bins')
            axes1[4].tick_params(axis='x', rotation=45)
            axes1[4].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # Plot 6: Length change distribution by time bins
            sns.boxplot(data=movement_df, x=time_bins, y='length_change', 
                    ax=axes1[5], palette='viridis')
            axes1[5].set_title('Length Change Distribution Across Time Bins')
            axes1[5].set_ylabel('Length Change (microns)')
            axes1[5].set_xlabel(f'Time ({time_label}s) Bins')
            axes1[5].tick_params(axis='x', rotation=45)
            axes1[5].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'temporal_progression_{resolution}_{location_type}.png'),
                    bbox_inches='tight', dpi=300)
            plt.close()
            
            # FIGURE 2: RELATIONSHIP AND DISTRIBUTION PLOTS
            fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
            axes2 = axes2.flatten()
            
            # Plot 1: Velocity vs Length Change (EXPLAINED)
            for location_type in movement_df['location_type_detailed'].unique():
                loc_data = movement_df[movement_df['location_type_detailed'] == location_type]
                
                # Color by movement type for clarity
                advancement_data = loc_data[loc_data['movement_type'] == 'advancement']
                retraction_data = loc_data[loc_data['movement_type'] == 'retraction']
                
                axes2[0].scatter(advancement_data[velocity_col], advancement_data['length_change'],
                            alpha=0.7, label=f'{location_type} (Advance)', s=40, marker='o')
                axes2[0].scatter(retraction_data[velocity_col], retraction_data['length_change'],
                            alpha=0.7, label=f'{location_type} (Retract)', s=40, marker='s')
            
            axes2[0].set_xlabel('Signed Velocity (positive = advancement, negative = retraction)')
            axes2[0].set_ylabel('Length Change (positive = growth, negative = shrinkage)')
            axes2[0].set_title('HOW TO INTERPRET: Velocity vs Length Change\n' +
                            'Quadrant I (top-right): Fast advancement with growth\n' +
                            'Quadrant II (top-left): Fast retraction but still growing(?!)\n' +
                            'Quadrant III (bottom-left): Fast retraction with shrinkage\n' +
                            'Quadrant IV (bottom-right): Fast advancement but shrinking(?!)\n' +
                            'Points near axes: Movement without length change or vice versa')
            axes2[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes2[0].grid(True, alpha=0.3)
            axes2[0].axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
            axes2[0].axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            # Add quadrant labels
            axes2[0].text(0.7, 0.7, 'I: Advancing + Growing', transform=axes2[0].transAxes, 
                        fontsize=8, alpha=0.7, ha='center')
            axes2[0].text(0.3, 0.7, 'II: Retracting but Growing', transform=axes2[0].transAxes, 
                        fontsize=8, alpha=0.7, ha='center')
            axes2[0].text(0.3, 0.3, 'III: Retracting + Shrinking', transform=axes2[0].transAxes, 
                        fontsize=8, alpha=0.7, ha='center')
            axes2[0].text(0.7, 0.3, 'IV: Advancing but Shrinking', transform=axes2[0].transAxes, 
                        fontsize=8, alpha=0.7, ha='center')
            
            # Plot 2: Distribution plots by movement type
            movement_types = movement_df['movement_type'].unique()
            colors = ['green', 'red']  # Green for advancement, red for retraction
            
            for i, mov_type in enumerate(movement_types):
                mov_data = movement_df[movement_df['movement_type'] == mov_type]
                axes2[1].hist(mov_data[velocity_col], bins=30, alpha=0.7, 
                            color=colors[i], label=mov_type, density=True)
            
            axes2[1].set_xlabel('Signed Velocity')
            axes2[1].set_ylabel('Density')
            axes2[1].set_title('Velocity Distribution by Movement Type')
            axes2[1].legend()
            axes2[1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            axes2[1].grid(True, alpha=0.3)
            
            # Plot 3: Correlation heatmap of movement features
            correlation_cols = [velocity_col, 'length_change', 'net_displacement', 
                            'displacement_magnitude']
            corr_data = movement_df[correlation_cols].corr()
            
            im = axes2[2].imshow(corr_data, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            axes2[2].set_xticks(range(len(correlation_cols)))
            axes2[2].set_yticks(range(len(correlation_cols)))
            axes2[2].set_xticklabels(correlation_cols, rotation=45)
            axes2[2].set_yticklabels(correlation_cols)
            axes2[2].set_title('Feature Correlation Heatmap')
            
            # Add correlation values as text
            for i in range(len(correlation_cols)):
                for j in range(len(correlation_cols)):
                    axes2[2].text(j, i, f'{corr_data.iloc[i, j]:.2f}', 
                                ha='center', va='center', fontsize=8)
            
            # Plot 4: Movement type proportion by location
            movement_proportions = movement_df.groupby(['location_type_detailed', 'movement_type']).size().unstack(fill_value=0)
            movement_percentages = movement_proportions.div(movement_proportions.sum(axis=1), axis=0) * 100
            
            movement_percentages.plot(kind='bar', ax=axes2[3], stacked=True, 
                                    color=['green', 'red'])
            axes2[3].set_title('Movement Type Proportions by Location')
            axes2[3].set_ylabel('Percentage (%)')
            axes2[3].set_xlabel('Location Type')
            axes2[3].tick_params(axis='x', rotation=45)
            axes2[3].legend(title='Movement Type')
            axes2[3].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'relationships_distributions_{resolution}_{location_type}.png'),
                    bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Saved {resolution}-level movement dynamics plots (2 figures)")
            
        except Exception as e:
            print(f"Error plotting movement dynamics: {e}")
    
    def generate_comprehensive_report(self) -> None:
        """Generate a comprehensive analysis report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE NEURITE ANALYSIS REPORT")
        print("="*60)
        
        # Generate all analyses
        self.plot_global_id_trajectories_with_means()
        self.analyze_temporal_features_high_res()
        
        print(f"\nAnalysis complete! All outputs saved to: {self.output_dir}")
        print("\nGenerated files:")
        for file in os.listdir(self.output_dir):
            if file.endswith('.png') or file.endswith('.csv'):
                file_path = os.path.join(self.output_dir, file)
                file_size = os.path.getsize(file_path) / 1024  # Size in KB
                print(f"  - {file} ({file_size:.1f} KB)")

def main():
    """Main function to run the complete analysis for multiple CSV files."""
    try:
        # Define the base directory containing all your data
        base_data_dir = "data/outputs_by_location_new_clustering"
        
        # Find all CSV files in the directory structure
        csv_pattern = os.path.join(base_data_dir, "**", "clustered_data_*.csv")
        csv_files = glob.glob(csv_pattern, recursive=True)
        
        if not csv_files:
            print(f"No CSV files found matching pattern: {csv_pattern}")
            return
        
        print(f"Found {len(csv_files)} CSV files to process:")
        for csv_file in csv_files:
            print(f"  - {csv_file}")
        
        # Process each CSV file
        for csv_path in csv_files:
            try:
                print(f"\n{'='*60}")
                print(f"Processing: {csv_path}")
                print(f"{'='*60}")
                
                # Extract the dataset name from the file path
                # Example: "clustered_data_mushroom_p4.csv" -> "mushroom_p4"
                file_name = Path(csv_path).stem  # Gets filename without extension
                dataset_name = file_name.replace('clustered_data_', '')
                
                # Create output directory based on dataset name
                output_dir = f"data/results_per_csv/{dataset_name}"
                os.makedirs(output_dir, exist_ok=True)
                
                # Initialize analyzer and run analysis
                analyzer = NeuriteAnalyzer(csv_path, output_dir)
                analyzer.generate_comprehensive_report()
                
                print(f"✓ Successfully processed {dataset_name}")
                print(f"  Results saved to: {output_dir}")
                
            except Exception as e:
                print(f"✗ Error processing {csv_path}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print("All processing completed!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error in main analysis: {e}")

def join_all_neurite_data():
    """Join all clustered_data CSV files into a single combined dataset."""
    try:
        base_data_dir = "data/outputs_by_location_vids"
        csv_pattern = os.path.join(base_data_dir, "**", "clustered_data_*.csv")
        csv_files = glob.glob(csv_pattern, recursive=True)
        
        if not csv_files:
            print("No CSV files found to join.")
            return None
        
        all_dataframes = []
        
        for csv_path in csv_files:
            try:
                # Extract dataset name
                file_name = Path(csv_path).stem
                dataset_name = file_name.replace('clustered_data_', '')
                
                print(f"Reading: {dataset_name}")
                df = pd.read_csv(csv_path)
                
                # Add dataset identifier column
                df['dataset'] = dataset_name
                
                # Handle potential ID conflicts by creating unique IDs
                if 'id' in df.columns:
                    df['original_id'] = df['id']  # Keep original ID for reference
                    df['unique_id'] = df['dataset'] + '_' + df['id'].astype(str)
                    df['id'] = df['unique_id']  # Replace ID with unique version
                
                all_dataframes.append(df)
                print(f"✓ Added {len(df)} rows from {dataset_name}")
                
            except Exception as e:
                print(f"✗ Error reading {csv_path}: {e}")
                continue
        
        if not all_dataframes:
            print("No dataframes were successfully read.")
            return None
        
        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Save combined dataset
        output_path = "data/combined_neurite_analysis/all_datasets_combined.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        
        print(f"\n{'='*60}")
        print("SUCCESS: All datasets combined!")
        print(f"Total rows: {len(combined_df)}")
        print(f"Total datasets: {len(all_dataframes)}")
        print(f"Saved to: {output_path}")
        print(f"{'='*60}")
        
        # Print dataset summary
        print("\nDataset summary:")
        summary = combined_df['dataset'].value_counts()
        for dataset, count in summary.items():
            print(f"  {dataset}: {count} rows")
        
        return combined_df
        
    except Exception as e:
        print(f"Error joining neurite data: {e}")
        return None

def main_with_combined_analysis():
    """Run analysis on both individual datasets and combined dataset."""
    try:
        # First, process individual datasets
        base_data_dir = "data/outputs_by_location_vids"
        csv_pattern = os.path.join(base_data_dir, "**", "clustered_data_*.csv")
        csv_files = glob.glob(csv_pattern, recursive=True)
    
        
        # Then create and analyze combined dataset
        print(f"\n{'='*60}")
        print("CREATING COMBINED DATASET")
        print(f"{'='*60}")
        
        combined_df = join_all_neurite_data()
        
        if combined_df is not None:
            # Analyze combined dataset
            combined_output_dir = "data/combined_neurite_analysis"
            os.makedirs(combined_output_dir, exist_ok=True)
            
            # Save the combined dataframe temporarily for analysis
            temp_combined_path = os.path.join(combined_output_dir, "temp_combined_data.csv")
            combined_df.to_csv(temp_combined_path, index=False)
            
            print(f"\nAnalyzing combined dataset...")
            combined_analyzer = NeuriteAnalyzer(temp_combined_path, combined_output_dir)
            combined_analyzer.generate_comprehensive_report()
            
            # Clean up temporary file if desired
            os.remove(temp_combined_path)
            
        print("\nAll processing completed!")
        
    except Exception as e:
        print(f"Error in combined analysis: {e}")
        
def main_single_file():
    """Alternative main function to process a single specific file."""
    try:
        # Choose one of these paths
        data_paths = [
            "data/outputs_by_location_new_clustering/mushroom_p4/pngs/clustered_data_mushroom_p4.csv",
            "data/outputs_by_location_new_clustering/mushroom_p10/pngs/clustered_data_mushroom_p10.csv", 
            "data/outputs_by_location_new_clustering/thin_p4_p10/pngs/clustered_data_thin_p4_p10.csv"
        ]
        
        # Select which file to process (0, 1, or 2)
        selected_index = 2  # Change this index to process different files
        
        data_path = data_paths[selected_index]
        
        # Extract dataset name from path
        file_name = Path(data_path).stem
        dataset_name = file_name.replace('clustered_data_', '')
        
        output_dir = f"data/results_per_csv/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing: {data_path}")
        print(f"Output directory: {output_dir}")
        
        analyzer = NeuriteAnalyzer(data_path, output_dir)
        analyzer.generate_comprehensive_report()
        
        print(f"✓ Analysis completed for {dataset_name}")
        
    except Exception as e:
        print(f"Error in single file analysis: {e}")

def main_with_custom_list():
    """Main function with a custom list of specific files to process."""
    try:
        # Define specific files you want to process
        files_to_process = [
            "data/outputs_by_location_new_clustering/mushroom_p4/pngs/clustered_data_mushroom_p4.csv",
            "data/outputs_by_location_new_clustering/thin_p4_p10/pngs/clustered_data_thin_p4_p10.csv"
        ]
        
        for data_path in files_to_process:
            try:
                if not os.path.exists(data_path):
                    print(f"File not found: {data_path}")
                    continue
                
                # Extract dataset name
                file_name = Path(data_path).stem
                dataset_name = file_name.replace('clustered_data_', '')
                
                output_dir = f"data/results_per_csv/{dataset_name}"
                os.makedirs(output_dir, exist_ok=True)
                
                print(f"\nProcessing: {dataset_name}")
                print(f"Input: {data_path}")
                print(f"Output: {output_dir}")
                
                analyzer = NeuriteAnalyzer(data_path, output_dir)
                analyzer.generate_comprehensive_report()
                
                print(f"✓ Completed: {dataset_name}")
                
            except Exception as e:
                print(f"✗ Error processing {data_path}: {e}")
                continue
        
        print("\nCustom processing completed!")
        
    except Exception as e:
        print(f"Error in custom list analysis: {e}")

if __name__ == "__main__":
    # Choose which main function to run:
    # Option 1: Just combine data without individual analysis
    join_all_neurite_data()    
    
    # Option 2: Run complete analysis (individual + combined)
    main_with_combined_analysis()
    
    # Option 1: Process all CSV files automatically
    # main()
    
    # Option 2: Process a single specific file
    # main_single_file()
    
    # Option 3: Process a custom list of files
    # main_with_custom_list()