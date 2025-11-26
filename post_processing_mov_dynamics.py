import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Optional
import warnings
import glob
from pathlib import Path
from scipy import interpolate
import re

warnings.filterwarnings('ignore')

class MovementDataPlotter:
    """
    A comprehensive plotting class for analyzing and visualizing neurite movement dynamics data.
    
    This class provides methods to generate various types of plots including velocity analysis,
    displacement tracking, polar direction plots, and temporal feature analysis for studying
    neurite dynamics across different experimental conditions and location types.
    
    Attributes
    ----------
    data_path : str
        Path to the CSV file containing movement dynamics data
    output_dir : str
        Directory where generated plots will be saved
    df : pandas.DataFrame
        Loaded movement dynamics data
    """
    def __init__(self, data_path: str, output_dir: str = "movement_plots_enhanced"):
        """
        Initialize the MovementDataPlotter with data path and output directory.
        
        Parameters
        ----------
        data_path : str
            Path to the CSV file containing movement dynamics data
        output_dir : str, optional
            Directory where generated plots will be saved, by default "movement_plots_enhanced"
            
        Raises
        ------
        FileNotFoundError
            If the specified data file does not exist
        Exception
            For data loading or processing errors
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.create_output_directory()
        self.load_data()
        
    def create_output_directory(self) -> None:
        """
        Create the output directory for saving plots if it doesn't exist.
        
        Returns
        -------
        None
            Creates directory structure and provides confirmation message
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
    
    def load_data(self) -> None:
        """
        Load movement dynamics data from CSV file and perform initial validation.
        
        Returns
        -------
        None
            Loads data into self.df and provides summary statistics
            
        Raises
        ------
        Exception
            If data loading fails or required columns are missing
        """
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Movement data loaded: {len(self.df)} rows")
            print(f"Columns: {list(self.df.columns)}")
            print(f"Location types: {self.df['location_type_detailed'].unique()}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def remove_outliers(self, data: pd.Series, method: str = 'iqr', factor: float = 1.5) -> pd.Series:
        """Remove outliers from data using specified method."""
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            return data[(data >= lower_bound) & (data <= upper_bound)]
        elif method == 'std':
            mean = data.mean()
            std = data.std()
            return data[(data >= mean - factor*std) & (data <= mean + factor*std)]
        else:
            return data
    
    def plot_enhanced_velocity_analysis(self, movement_df, resolution) -> None:
        """Create enhanced velocity plots with better location type differentiation."""
        try:
            # Define the same color scheme as polar plots
            color_scheme = {
                'control': 'darkblue',
                'thin': 'orange',
                'mushroom': 'purple', 
                'stubby': 'green'
            }
            
            # Convert resolution to actual column name
            if resolution == 'minute':
                time_col = 'minute_start'
                velocity_col = 'signed_velocity_um_per_minute'
                velocity_label = 'Velocity (μm/minute)'
                title_suffix = 'per Minute'
            elif resolution == 'hour':
                time_col = 'hour_start'
                velocity_col = 'signed_velocity_um_per_hour' 
                velocity_label = 'Velocity (μm/hour)'
                title_suffix = 'per Hour'
            else:
                print(f"Unknown resolution: {resolution}")
                return
                
            print(f"Using time column: {time_col}")
            print(f"Using velocity column: {velocity_col}")
            
            # Check if the required columns exist in the dataframe
            if time_col not in movement_df.columns:
                print(f"Warning: Time column '{time_col}' not found in dataframe")
                print(f"Available columns: {[col for col in movement_df.columns if 'time' in col.lower() or 'start' in col.lower()]}")
                return
            
            if velocity_col not in movement_df.columns:
                print(f"Warning: Velocity column '{velocity_col}' not found in dataframe")
                print(f"Available velocity columns: {[col for col in movement_df.columns if 'velocity' in col.lower() or 'speed' in col.lower()]}")
                # Try to find alternative column names
                alt_velocity_cols = [col for col in movement_df.columns if 'velocity' in col.lower() or 'speed' in col.lower()]
                if alt_velocity_cols:
                    print(f"Available velocity-related columns: {alt_velocity_cols}")
                    # Use the first available velocity column as fallback
                    velocity_col = alt_velocity_cols[0]
                    print(f"Using alternative velocity column: {velocity_col}")
                else:
                    print("No velocity columns found. Cannot generate plots.")
                    return
                
            if 'location_type_detailed' not in movement_df.columns:
                print(f"Warning: 'location_type_detailed' column not found in dataframe")
                print(f"Available columns: {movement_df.columns.tolist()}")
                return
                
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            fig.suptitle('Comprehensive Velocity Analysis by Location Type', fontsize=16, fontweight='bold')
            
            # Remove outliers - FIXED: Use movement_df directly
            velocity_data = self.remove_outliers(movement_df[velocity_col], method='iqr', factor=2.0)
            # Create filtered dataframe using the original indices from outlier removal
            filtered_df = movement_df.loc[velocity_data.index].copy()
            
            # Get location types and order them by p-value
            all_location_types = filtered_df['location_type_detailed'].unique()
            
            # Separate control and experimental groups
            control_groups = [lt for lt in all_location_types if 'control' in lt.lower()]
            experimental_groups = [lt for lt in all_location_types if 'control' not in lt.lower()]
            
            # Sort experimental groups by shape and p-value
            def extract_pvalue(group_name):
                """Extract p-value from group name for sorting."""
                import re
                match = re.search(r'_p(\d+)', group_name)
                return int(match.group(1)) if match else 0
            
            # Group experimental data by shape
            shape_groups = {}
            for group in experimental_groups:
                base_shape = group.split('_')[0]  # Get "thin", "mushroom", etc.
                if base_shape not in shape_groups:
                    shape_groups[base_shape] = []
                shape_groups[base_shape].append(group)
            
            # Sort each shape group by p-value
            for shape in shape_groups:
                shape_groups[shape].sort(key=extract_pvalue)
            
            # Create ordered location types: control first, then experimental groups by shape and p-value
            location_types = control_groups.copy()  # Start with control
            
            # Add experimental groups in order: thin, mushroom, stubby
            for shape in ['thin', 'mushroom', 'stubby']:
                if shape in shape_groups:
                    location_types.extend(shape_groups[shape])
            
            print(f"Ordered location types: {location_types}")
            
            # Assign colors based on the color scheme
            colors = []
            for location_type in location_types:
                if 'control' in location_type.lower():
                    colors.append(color_scheme['control'])
                else:
                    base_shape = location_type.split('_')[0]
                    colors.append(color_scheme.get(base_shape, 'gray'))
            
            print(f"Velocity statistics after outlier removal:")
            print(f"  Min: {velocity_data.min():.4f}")
            print(f"  Max: {velocity_data.max():.4f}")
            print(f"  Mean: {velocity_data.mean():.4f}")
            print(f"  Std: {velocity_data.std():.4f}")
            
            # --- PLOT 1: Velocity Over Time with Better Smoothing ---
            for i, location_type in enumerate(location_types):
                loc_data = filtered_df[filtered_df['location_type_detailed'] == location_type]
                
                if not loc_data.empty and len(loc_data) > 5:
                    sorted_data = loc_data.sort_values(time_col)
                    
                    # Use larger window for meaningful smoothing (10% of data points, min 5)
                    window_size = max(5, min(20, len(sorted_data) // 10))
                    rolling_avg = sorted_data[velocity_col].rolling(window=window_size, center=True, min_periods=1).mean()
                    
                    # Add confidence interval
                    rolling_std = sorted_data[velocity_col].rolling(window=window_size, center=True, min_periods=1).std()
                    
                    axes[0, 0].plot(sorted_data[time_col], rolling_avg,
                                color=colors[i], linewidth=2.5, label=location_type, marker='o', markersize=3)
                    axes[0, 0].fill_between(sorted_data[time_col], 
                                        rolling_avg - rolling_std, 
                                        rolling_avg + rolling_std,
                                        color=colors[i], alpha=0.2)
            
            axes[0, 0].set_xlabel('Time (minutes)')
            axes[0, 0].set_ylabel(velocity_label)
            axes[0, 0].set_title(f'A) Velocity Trends Over Time\n(Rolling Mean ± STD) {title_suffix}')
            axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            # --- IMPROVED PLOT 2: Comparative Distribution Analysis ---
            # Create violin + box plot combination for better comparison
            plot_data = []
            for location_type in location_types:
                loc_data = filtered_df[filtered_df['location_type_detailed'] == location_type]
                if not loc_data.empty:
                    for velocity in loc_data[velocity_col]:
                        plot_data.append({'location_type': location_type, 'velocity': velocity})
            
            plot_df = pd.DataFrame(plot_data)
            
            if not plot_df.empty:
                # Violin plot shows distribution shape
                parts = axes[0, 1].violinplot([plot_df[plot_df['location_type'] == lt]['velocity'] 
                                            for lt in location_types],
                                            showmeans=True, showmedians=True)
                
                # Color the violins using our color scheme
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.7)
                
                # Add individual data points with jitter
                for i, location_type in enumerate(location_types):
                    loc_velocities = plot_df[plot_df['location_type'] == location_type]['velocity']
                    if len(loc_velocities) > 0:
                        x = np.random.normal(i+1, 0.1, size=len(loc_velocities))
                        axes[0, 1].scatter(x, loc_velocities, alpha=0.4, color=colors[i], s=20)
            
            axes[0, 1].set_xlabel('Location Type')
            axes[0, 1].set_ylabel(velocity_label)
            axes[0, 1].set_title(f'B) Velocity Distribution Comparison\n(Violin + Scatter Plot) {title_suffix}')
            axes[0, 1].set_xticks(range(1, len(location_types) + 1))
            axes[0, 1].set_xticklabels(location_types, rotation=45)
            axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[0, 1].grid(True, alpha=0.3)
            
            # --- PLOT 3: Cumulative Velocity with Statistical Comparison ---
            max_time = filtered_df[time_col].max()
            for i, location_type in enumerate(location_types):
                loc_data = filtered_df[filtered_df['location_type_detailed'] == location_type].sort_values(time_col)
                
                if not loc_data.empty:
                    cumulative_velocity = loc_data[velocity_col].cumsum()
                    axes[1, 0].plot(loc_data[time_col], cumulative_velocity,
                                color=colors[i], linewidth=2.5, label=location_type)
                    
                    # Add final displacement annotation
                    final_disp = cumulative_velocity.iloc[-1]
                    axes[1, 0].annotate(f'{final_disp:.1f}μm', 
                                    xy=(max_time, final_disp),
                                    xytext=(10, 0), textcoords='offset points',
                                    fontweight='bold', color=colors[i])
            
            axes[1, 0].set_xlabel('Time (minutes)')
            axes[1, 0].set_ylabel('Cumulative Displacement (μm)')
            axes[1, 0].set_title(f'C) Net Displacement Over Time\n(Total movement integral) {title_suffix}')
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            # --- IMPROVED PLOT 4: Location Type Comparison Statistics ---
            stats_data = []
            for location_type in location_types:
                loc_data = filtered_df[filtered_df['location_type_detailed'] == location_type]
                if not loc_data.empty:
                    stats_data.append({
                        'location_type': location_type,
                        'mean_velocity': loc_data[velocity_col].mean(),
                        'std_velocity': loc_data[velocity_col].std(),
                        'median_velocity': loc_data[velocity_col].median(),
                        'count': len(loc_data)
                    })
            
            stats_df = pd.DataFrame(stats_data)
            
            if not stats_df.empty:
                x_pos = range(len(stats_df))
                bars = axes[1, 1].bar(x_pos, stats_df['mean_velocity'], 
                                    yerr=stats_df['std_velocity'],
                                    capsize=5, alpha=0.7,
                                    color=colors[:len(stats_df)])
                
                # Add value labels on bars
                for i, (idx, row) in enumerate(stats_df.iterrows()):
                    axes[1, 1].text(i, row['mean_velocity'] + (row['std_velocity'] if not np.isnan(row['std_velocity']) else 0),
                                f'{row["mean_velocity"]:.2f}\n(n={row["count"]})',
                                ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            axes[1, 1].set_xlabel('Location Type')
            axes[1, 1].set_ylabel(f'Mean Velocity {title_suffix}')
            axes[1, 1].set_title(f'D) Velocity Statistics by Location Type\n(Mean ± STD) {title_suffix}')
            axes[1, 1].set_xticks(range(len(stats_df)))
            axes[1, 1].set_xticklabels(stats_df['location_type'], rotation=45)
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'enhanced_velocity_analysis_{resolution}.png'),
                    bbox_inches='tight', dpi=300)
            plt.savefig(os.path.join(self.output_dir, f'enhanced_velocity_analysis_{resolution}.pdf'),
                    bbox_inches='tight', dpi=300)
            plt.close()
            
            # Print statistical comparison
            print(f"\n--- Statistical Comparison by Location Type ({resolution}) ---")
            for location_type in location_types:
                loc_data = filtered_df[filtered_df['location_type_detailed'] == location_type]
                if not loc_data.empty:
                    print(f"{location_type}: mean={loc_data[velocity_col].mean():.4f}, "
                        f"std={loc_data[velocity_col].std():.4f}, "
                        f"n={len(loc_data)}")
            
            print(f"Saved enhanced velocity analysis plot for {resolution}")
            
        except Exception as e:
            print(f"Error in enhanced velocity analysis: {e}")
            import traceback
            traceback.print_exc()
            
    def plot_microns_displacement_analysis(self, resolution) -> None:
        """Plot displacement analysis in microns with line plots and polar plots."""
        try:
            
            # Get displacement columns in microns
            displacement_col = 'net_displacement'
            velocity_col = 'signed_velocity'
            
            # Convert resolution to actual column name
            if resolution == 'minute':
                velocity_col = 'signed_velocity_um_per_minute'
                velocity_label = 'Velocity (μm/minute)'
                title_suffix = 'per Minute'
                time_col = 'minute_start'
            elif resolution == 'hour':  # Default to hour
                velocity_col = 'signed_velocity_um_per_hour' 
                velocity_label = 'Velocity (μm/hour)'
                title_suffix = 'per Hour'
                time_col = 'hour_start'    
            else:
                print(f"Unknown resolution: {resolution}")
                return
                
            print(f"Using time column: {time_col}")
            
            # Check if the time column exists in the dataframe
            if time_col not in self.df.columns:
                print(f"Warning: Time column '{time_col}' not found in dataframe")
                print(f"Available columns: {[col for col in self.df.columns if 'time' in col.lower() or 'start' in col.lower()]}")
                return
            
            # Create output directory for this analysis
            analysis_dir = os.path.join(self.output_dir, 'displacement_analysis')
            if not os.path.exists(analysis_dir):
                os.makedirs(analysis_dir)
            
            # 1. TEMPORAL LINE PLOTS
            self._plot_temporal_line_plots(displacement_col, velocity_col, time_col, resolution, analysis_dir)
                        
            print("Saved all displacement analysis plots")
            
        except Exception as e:
            print(f"Error in displacement analysis: {e}")

    def _get_location_colors(self, location_types):
        """Return a color mapping for location types."""
        # Use matplotlib color cycle or define custom colors
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        # Create a mapping from location_type to color
        color_map = {}
        for i, loc_type in enumerate(location_types):
            color_map[loc_type] = color_cycle[i % len(color_cycle)]
        
        return color_map

    def _aggregate_by_time(self, data: pd.DataFrame, time_col: str, value_col: str) -> pd.DataFrame:
        """Aggregate data by time using mean."""
        return data.groupby(time_col)[value_col].mean().reset_index()

    def _plot_temporal_line_plots(self, displacement_col: str, velocity_col: str, time_col: str, resolution: str, output_dir: str) -> None:
        """Create temporal line plots for displacement and velocity with consistent colors per location type."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        location_types = sorted(self.df['location_type_detailed'].unique())
        colors = self._get_location_colors(location_types)
        
        # Validate that colors is a dictionary mapping location_type to color
        if not isinstance(colors, dict):
            # If _get_location_colors returns a list, convert to dict
            if isinstance(colors, list) and len(colors) == len(location_types):
                colors = dict(zip(location_types, colors))
            else:
                # Fallback: generate a color cycle
                color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                colors = {loc_type: color_cycle[i % len(color_cycle)] 
                        for i, loc_type in enumerate(location_types)}
        
        # Plot 1: Net Displacement Over Time (Raw)
        for location_type in location_types:
            loc_data = self.df[self.df['location_type_detailed'] == location_type]
            if not loc_data.empty:
                # Aggregate by time
                time_agg = self._aggregate_by_time(loc_data, time_col, displacement_col)
                if len(time_agg) > 1:
                    axes[0, 0].plot(time_agg[time_col], time_agg[displacement_col],
                                color=colors[location_type], linewidth=2, 
                                label=location_type, alpha=0.8)
        
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('Net Displacement (μm)')
        axes[0, 0].set_title('Net Displacement Over Time\n(Positive = Advancement, Negative = Retraction)')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # Plot 2: Velocity Over Time
        for location_type in location_types:
            loc_data = self.df[self.df['location_type_detailed'] == location_type]
            if not loc_data.empty:
                time_agg = self._aggregate_by_time(loc_data, time_col, velocity_col)
                if len(time_agg) > 1:
                    axes[0, 1].plot(time_agg[time_col], time_agg[velocity_col],
                                color=colors[location_type], linewidth=2,
                                label=location_type, alpha=0.8)
        
        axes[0, 1].set_xlabel('Time (minutes)')
        axes[0, 1].set_ylabel('Velocity (μm/min)')
        axes[0, 1].set_title('Velocity Over Time\n(Positive = Advancement, Negative = Retraction)')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # Plot 3: Cumulative Net Displacement
        for location_type in location_types:
            loc_data = self.df[self.df['location_type_detailed'] == location_type].sort_values(time_col)
            if not loc_data.empty:
                cumulative_disp = loc_data[displacement_col].cumsum()
                axes[1, 0].plot(loc_data[time_col], cumulative_disp,
                            color=colors[location_type], linewidth=2,
                            label=location_type, alpha=0.8)
        
        axes[1, 0].set_xlabel('Time (minutes)')
        axes[1, 0].set_ylabel('Cumulative Displacement (μm)')
        axes[1, 0].set_title('Cumulative Net Displacement\n(Total Position Change)')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # Plot 4: Positive vs Negative Movement Balance
        for location_type in location_types:
            loc_data = self.df[self.df['location_type_detailed'] == location_type].sort_values(time_col)
            if not loc_data.empty:
                # Calculate rolling balance (positive - negative movements)
                positive_mask = loc_data[velocity_col] > 0
                negative_mask = loc_data[velocity_col] < 0
                
                if positive_mask.any():
                    positive_movements = loc_data[positive_mask][velocity_col].rolling(window=5, min_periods=1).mean()
                    time_positive = loc_data[positive_mask][time_col]
                    if len(positive_movements) > 1:
                        axes[1, 1].plot(time_positive, positive_movements,
                                    color=colors[location_type], linewidth=2, 
                                    label=f'{location_type} (Advance)', alpha=0.7)
                
                if negative_mask.any():
                    negative_movements = loc_data[negative_mask][velocity_col].rolling(window=5, min_periods=1).mean()
                    time_negative = loc_data[negative_mask][time_col]
                    if len(negative_movements) > 1:
                        axes[1, 1].plot(time_negative, negative_movements,
                                    color=colors[location_type], linewidth=2, 
                                    linestyle='--', label=f'{location_type} (Retract)', alpha=0.7)
        
        axes[1, 1].set_xlabel('Time (minutes)')
        axes[1, 1].set_ylabel('Velocity (μm/min)')
        axes[1, 1].set_title('Advancement vs Retraction Velocity\n(Solid = Advance, Dashed = Retract)')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'temporal_line_plots_{location_type}_{resolution}.png'),
                bbox_inches='tight', dpi=300)
        plt.close()
        
    def _create_polar_count_plot(self, ax, data, color, label):
        """Create non-normalized count-based polar plot for advancement/retraction."""
        if 'signed_direction_rad' not in data.columns:
            return
        
        directions = data['signed_direction_rad']
        
        # Create histogram of counts (not normalized)
        bin_edges = np.linspace(0, 2*np.pi, 37)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        counts, _ = np.histogram(directions, bins=bin_edges)
        
        # Convert to proportional frequency (but don't normalize to max=1)
        total_movements = len(data)
        frequencies = counts / total_movements if total_movements > 0 else counts
        
        # Create closed loop
        centers_loop = np.concatenate([bin_centers, [bin_centers[0]]])
        freq_loop = np.concatenate([frequencies, [frequencies[0]]])
        
        ax.plot(centers_loop, freq_loop, color=color, linewidth=2, label=label, alpha=0.9)

    def _create_polar_line_with_std(self, ax, data, value_col=None, weights=None, 
                                color='blue', label='', normalize=True):
        """Create polar line plot with standard deviation shading."""
        if 'signed_direction_rad' not in data.columns:
            return
        
        directions = data['signed_direction_rad']
        
        # Use values if provided, otherwise count each movement as 1
        if value_col is not None and value_col in data.columns:
            values = np.abs(data[value_col])  # Use magnitude for directional plots
        else:
            values = np.ones(len(data))
        
        # Apply weights if specified
        if weights is not None:
            values = values * weights
        
        # Create histogram
        bin_edges = np.linspace(0, 2*np.pi, 37)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        hist, _ = np.histogram(directions, bins=bin_edges, weights=values)
        count, _ = np.histogram(directions, bins=bin_edges)
        
        # Calculate mean and std for each bin
        bin_means = []
        bin_stds = []
        valid_centers = []
        
        for i in range(len(bin_edges) - 1):
            bin_mask = (directions >= bin_edges[i]) & (directions < bin_edges[i+1])
            if bin_mask.any():
                bin_values = values[bin_mask]
                bin_means.append(np.mean(bin_values))
                bin_stds.append(np.std(bin_values))
                valid_centers.append(bin_centers[i])
        
        if not valid_centers:
            return
        
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        valid_centers = np.array(valid_centers)
        
        # Normalize if requested - but only if we have variation
        if normalize and len(bin_means) > 0 and np.max(bin_means) > 0:
            max_val = np.max(bin_means)
            bin_means = bin_means / max_val
            bin_stds = bin_stds / max_val
        
        # Create closed loop
        centers_loop = np.concatenate([valid_centers, [valid_centers[0]]])
        means_loop = np.concatenate([bin_means, [bin_means[0]]])
        stds_loop = np.concatenate([bin_stds, [bin_stds[0]]])
        
        # Plot
        ax.plot(centers_loop, means_loop, color=color, linewidth=2, label=label, alpha=0.9)
        ax.fill_between(centers_loop, 
                    np.maximum(0, means_loop - stds_loop),  # Don't go below 0
                    means_loop + stds_loop, 
                    color=color, alpha=0.3)
    
    
    def _filter_data_for_plot(self, data, title, value_col, resolution):
        """Filter data based on plot title and parameters."""
        filtered_data = data.copy()
        
        if resolution == 'minute':
            velocity_col = 'signed_velocity_um_per_minute'
            velocity_label = 'Velocity (μm/minute)'
            title_suffix = 'per Minute'
        elif resolution == 'hour':  # Default to hour
            velocity_col = 'signed_velocity_um_per_hour' 
            velocity_label = 'Velocity (μm/hour)'
            title_suffix = 'per Hour'
        else:
            print(f"Velocity column {velocity_col} not found, using hour data")
            velocity_col = 'signed_velocity_um_per_hour'
            velocity_label = 'Velocity (μm/hour)'
            title_suffix = 'per Hour'
        
        if "Advancement" in title and velocity_col in data.columns:
            filtered_data = filtered_data[filtered_data[velocity_col] > 0]
        elif "Retraction" in title and velocity_col in data.columns:
            filtered_data = filtered_data[filtered_data[velocity_col] <= 0]
        elif "Large Movements" in title and value_col in data.columns:
            threshold = filtered_data[value_col].quantile(0.50)
            filtered_data = filtered_data[filtered_data[value_col] >= threshold]
        elif "Small Movements" in title and value_col in data.columns:
            threshold = filtered_data[value_col].quantile(0.25)
            filtered_data = filtered_data[filtered_data[value_col] <= threshold]
        
        return filtered_data

    def _plot_polar_direction_analysis(self, time_col: str, output_dir: str) -> None:
        """Create polar plots showing movement direction preferences.
        
        Key Insights:
        - Angle: Movement direction (0° = right, 90° = up, etc.)
        - Radius: Frequency or magnitude of movements in that direction
        - Full circle = movements in all directions
        - Half circle = directional bias
        - Normalized radius (0-1) allows comparison between conditions
        """
        location_types = sorted(self.df['location_type_detailed'].unique())
        colors = self._get_location_colors(location_types)
        
        fig = plt.figure(figsize=(20, 16))
        
        # Revised plot configurations with better descriptions
        polar_configs = [
            # (position, title, value_col, weight_col, normalize, description)
            (331, "Movement Frequency by Direction\n(All Movements, Count-Based)", 
            None, None, True, "Shows which directions movements occur in, regardless of magnitude"),
            
            (332, "Velocity-Weighted Direction Preference\n(Fast Movements Emphasized)", 
            'signed_velocity_um_per_min', 'signed_velocity_um_per_min', True,
            "Directions weighted by speed - shows where FASTEST movements occur"),
            
            (333, "Advancement Direction Preference\n(Velocity > 0)", 
            None, None, False, "Raw count of forward movements by direction"),
            
            (334, "Retraction Direction Preference\n(Velocity < 0)", 
            None, None, False, "Raw count of backward movements by direction"),
            
            (335, "Movement Magnitude by Direction", 
            'displacement_magnitude', 'displacement_magnitude', True,
            "Directions weighted by distance traveled"),
            
            (336, "Net Displacement Vector Direction\n(Signed, Not Normalized)", 
            'net_displacement', 'net_displacement', False,
            "Actual displacement direction - positive radius = net advancement"),
            
            (337, "Large Movement Directions\n(Top 25% by Distance)", 
            'displacement_magnitude', None, True,
            "Only the largest 25% of movements by distance"),
            
            (338, "Small Movement Directions\n(Bottom 25% by Distance)", 
            'displacement_magnitude', None, True,
            "Only the smallest 25% of movements by distance"),
            
            (338, "Velocity-Weighted Direction Preference\n(Fast Movements Emphasized)", 
            'signed_velocity_um_per_hour', 'signed_velocity_um_per_hour', True,
            "Directions weighted by speed - shows where FASTEST movements occur"),
        ]
        
        for (position, title, value_col, weight_col, normalize, description) in polar_configs:
            ax = fig.add_subplot(position, projection='polar')
            
            for location_type in location_types:
                loc_data = self.df[self.df['location_type_detailed'] == location_type].copy()
                
                if not loc_data.empty:
                    filtered_data = self._filter_data_for_plot(loc_data, title, value_col, time_col)
                    
                    if not filtered_data.empty:
                        # Special handling for advancement/retraction to avoid full circles
                        if "Advancement" in title or "Retraction" in title:
                            self._create_polar_count_plot(ax, filtered_data, colors[location_type], location_type)
                        else:
                            weights = None
                            if weight_col and weight_col in filtered_data.columns:
                                weights = np.abs(filtered_data[weight_col])
                            
                            self._create_polar_line_with_std(
                                ax, filtered_data, 
                                value_col=value_col,
                                weights=weights, 
                                color=colors[location_type],
                                label=location_type,
                                normalize=normalize
                            )
            
            ax.set_title(f"{title}\n{description}", pad=20, size=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add radial labels to show it's normalized
            if normalize:
                ax.set_yticks([0, 0.5, 1.0])
                ax.set_yticklabels(['0', '0.5', '1.0 (max)'])
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], color=colors[lt], linewidth=3, label=lt) 
                        for lt in location_types]
        fig.legend(handles=legend_elements, loc='upper center', 
                bbox_to_anchor=(0.5, 0.02), ncol=min(4, len(location_types)), fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.savefig(os.path.join(output_dir, f'comprehensive_polar_analysis_{location_type}.png'),
                bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(output_dir, f'comprehensive_polar_analysis_{location_type}.pdf'),
                bbox_inches='tight', dpi=300)
        plt.close()
        
    def _plot_polar_directions(self, movement_df: pd.DataFrame, resolution: str) -> None:
        """Plot polar plots with signed directions preserving negative angles."""
        try:
            # Define color scheme
            color_scheme = {
                'control': 'darkblue',
                'thin': 'orange',
                'mushroom': 'purple', 
                'stubby': 'green'
            }
            
            # DEBUG: Check what's in the data
            print(f"\n🔍 DEBUG: Checking movement_df contents")
            print(f"Total rows: {len(movement_df)}")
            print(f"location_type unique values: {movement_df['location_type'].unique()}")
            print(f"location_type_detailed unique values: {movement_df['location_type_detailed'].unique()}")
            print(f"shape unique values: {movement_df['shape'].unique()}")
            
            # Identify control group - use location_type_detailed for consistency
            control_groups = [group for group in movement_df['location_type_detailed'].unique() 
                            if 'control' in group.lower()]
            
            if not control_groups:
                print("No control group found for polar plot comparison")
                return
            
            # Use the first control group found
            control_group = control_groups[0]
            print(f"Using control group: {control_group}")
            
            # Get experimental data (everything that's not control)
            experimental_data = movement_df[~movement_df['location_type_detailed'].isin(control_groups)]
            
            # Get unique shapes from experimental data
            shapes = experimental_data['shape'].unique()
            print(f"Found shapes: {shapes}")
            
            # Create ordered groups by shape and pvalue
            ordered_groups = []
            for shape in shapes:
                # Get all groups for this shape
                shape_groups = experimental_data[experimental_data['shape'] == shape]['location_type_detailed'].unique()
                print(f"Shape {shape} has groups: {list(shape_groups)}")
                
                # Extract pvalues and sort them (p4, p10, p30)
                pvalue_groups = []
                for group in shape_groups:
                    match = re.search(r'_p(\d+)', group)
                    if match:
                        pval = int(match.group(1))
                        pvalue_groups.append((pval, group))
                
                # Sort by pvalue and add to ordered groups
                pvalue_groups.sort(key=lambda x: x[0])
                for pval, group in pvalue_groups:
                    ordered_groups.append(group)
            
            # Add control at the beginning
            all_groups = [control_group] + ordered_groups
            print(f"All groups to plot: {all_groups}")
            
            # Create TWO types of plots:
            # 1. Individual plots for each group
            # 2. Combined plots with control + experimental groups
            
            # PLOT 1: Individual plots for each group
            self._create_individual_polar_plots(movement_df, resolution, all_groups, color_scheme)
            
            # PLOT 2: Combined plots - control vs each shape type
            self._create_combined_polar_plots(movement_df, resolution, color_scheme, control_group)
            
            print(f"Saved polar direction plots for {resolution}")
            
        except Exception as e:
            print(f"Error creating polar direction plots: {e}")
            import traceback
            traceback.print_exc()

    def _create_individual_polar_plots(self, movement_df: pd.DataFrame, resolution: str, groups: list, color_scheme: dict) -> None:
        """Create individual polar plots for each group."""
        # Create figure with appropriate number of subplots
        n_groups = len(groups)
        n_cols = min(3, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols
        # Define fixed radial limits for all subplots
        # Define fixed radial limits for all subplots
        FIXED_MAX_RADIUS = 27.0
        RADIAL_TICKS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]  # Consistent tick marks


        fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
        
        for i, group in enumerate(groups, 1):
            if i > n_rows * n_cols:
                break
                    
            ax = fig.add_subplot(n_rows, n_cols, i, projection='polar')
            
            # ALWAYS use location_type_detailed for consistency
            group_data = movement_df[movement_df['location_type_detailed'] == group]
            
            print(f"📊 Plotting group: {group} - Found {len(group_data)} rows")
            
            # Determine color and label
            if 'control' in group.lower():
                group_label = 'Control'
                color = color_scheme['control']
            else:
                group_label = group
                # Extract shape from group name to get color
                shape = group.split('_')[0]
                color = color_scheme.get(shape, 'gray')  # Default to gray if shape not found
            
            if not group_data.empty:
                theta_data = group_data['signed_direction_rad'].values
                magnitude_data = group_data['displacement_magnitude'].values
    
                # Clip extreme values for better visualization
                magnitude_data = np.clip(magnitude_data, 0, FIXED_MAX_RADIUS)
                
                print(f"  Group {group}: {len(theta_data)} data points, magnitude range: {magnitude_data.min():.2f}-{magnitude_data.max():.2f} μm")
                
                # Create finer bins for smoother interpolation
                n_bins = 72  # Double the bins for smoother curves
                bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                # Set fixed radial limits for ALL plots
                ax.set_ylim(0, FIXED_MAX_RADIUS)
                ax.set_yticks(RADIAL_TICKS)
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
                valid_count = np.sum(valid_mask)
                print(f"  Group {group}: {valid_count}/{len(valid_mask)} bins have data")
                
                if valid_count >= 3:  # Need at least 3 points for interpolation
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
                
                # Plot with group-specific color
                ax.fill_between(theta_fine, lower_fine, upper_fine, 
                            alpha=0.3, color=color, label='68% CI')
                ax.plot(theta_fine, mean_fine, '-', linewidth=2, color=color, label='Mean Magnitude')
                
                # [Rest of the plotting code remains the same...]
                ax.set_theta_offset(np.pi/2)
                ax.set_theta_direction(-1)
                ax.set_thetamin(-180)
                ax.set_thetamax(180)
                ax.set_xticks(np.arange(-180, 181, 45) * np.pi/180)
                ax.set_xticklabels(['-180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°', '180°'])
                ax.set_yticklabels([])

                # Add quadrant lines and labels
                quadrant_angles = [-np.pi, -np.pi/2, 0, np.pi/2]
                quadrant_labels = ['Left (-180°)', 'Down (-90°)', 'Right (0°)', 'Up (90°)']
                quadrant_colors = ['red', 'purple', 'green', 'blue']
                
                for angle, label, qcolor in zip(quadrant_angles, quadrant_labels, quadrant_colors):
                    ax.plot([angle, angle], [0, FIXED_MAX_RADIUS  * 0.8], 
                        color=qcolor, linestyle='--', alpha=0.5, linewidth=1)
                    label_radius = FIXED_MAX_RADIUS * 0.9
                    ax.text(angle, label_radius, label, 
                        ha='center', va='center', 
                        fontsize=8, color=qcolor, 
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                
                ax.set_title(f'{group_label}\n(n={len(group_data)} movements)', pad=20, fontsize=12)
                ax.grid(True, alpha=0.3)
                
                
                if i == 1:
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                else:
                    coverage = valid_count / len(valid_mask) * 100
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
        plt.savefig(os.path.join(self.output_dir, f'polar_directions_individual_{resolution}.png'),
                bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(self.output_dir, f'polar_directions_individual_{resolution}.pdf'),
                bbox_inches='tight', dpi=300)
        plt.close()

    def _create_combined_polar_plots(self, movement_df: pd.DataFrame, resolution: str, color_scheme: dict, control_group) -> None:
        """Create combined polar plots showing control vs experimental groups."""
        try:
            # Get control data - FIXED: use location_type_detailed instead of location_type
            control_data = movement_df[movement_df['location_type_detailed'] == control_group]
            if control_data.empty:
                print(f"No control data found for {control_group}")
                return
            
            # Group experimental data by shape - FIXED: exclude control groups properly
            experimental_data = movement_df[~movement_df['location_type_detailed'].isin([control_group])]
            shapes = experimental_data['shape'].unique()
            
            print(f"Creating combined plots for shapes: {shapes}")
            
            # Create one combined plot per shape type
            for shape in shapes:
                # Get all groups for this shape from location_type_detailed
                shape_groups = experimental_data[experimental_data['shape'] == shape]['location_type_detailed'].unique()
                shape_groups = [group for group in shape_groups if shape in group]
                
                if not shape_groups:
                    print(f"No groups found for shape: {shape}")
                    continue
                    
                print(f"Processing shape {shape}: {shape_groups}")
                
                # Sort groups by pvalue (p4, p10, p30)
                def extract_pvalue(group_name):
                    match = re.search(r'_p(\d+)', group_name)
                    return int(match.group(1)) if match else 0
                
                shape_groups.sort(key=extract_pvalue)
                
                # Create figure for this shape
                n_plots = len(shape_groups)
                n_cols = min(2, n_plots)
                n_rows = (n_plots + n_cols - 1) // n_cols
                # Define fixed radial limits for all subplots
                FIXED_MAX_RADIUS = 27.0
                RADIAL_TICKS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]  # Consistent tick marks
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), 
                                    subplot_kw=dict(projection='polar'))
                
                # Handle single subplot case
                if n_plots == 1:
                    axes = [axes] if not hasattr(axes, '__len__') else axes
                else:
                    axes = axes.flatten()
                
                # Plot control vs each experimental group
                for i, (ax, exp_group) in enumerate(zip(axes, shape_groups)):
                    if i >= len(axes):
                        break
                        
                    # Get experimental group data
                    exp_data = movement_df[movement_df['location_type_detailed'] == exp_group]
                    
                    # FIXED: Get the base shape name for color lookup
                    base_shape = exp_group.split('_')[0]  # Get "thin" from "thin_p4"
                    exp_color = color_scheme.get(base_shape, 'gray')
                    
                    print(f"  Plotting {exp_group} (color: {exp_color}): {len(exp_data)} movements")
                    
                    # Clear the axis first
                    ax.clear()
                    
                    # Plot control (always in dark blue)
                    if not control_data.empty:
                        self._plot_single_group(ax, control_data, 'Control', color_scheme['control'], is_experimental=False, max_radius= FIXED_MAX_RADIUS)
                    
                    # Plot experimental group
                    if not exp_data.empty:
                        self._plot_single_group(ax, exp_data, exp_group, exp_color, is_experimental=True, max_radius=FIXED_MAX_RADIUS)
                    
                    # Set fixed radial limits for ALL plots
                    # After setting radial limits, add this to enforce clipping:
                    ax.set_ylim(0, FIXED_MAX_RADIUS)
                    ax.set_yticks(RADIAL_TICKS)

                    # ENFORCE CLIPPING at the axis level
                    ax.set_rlim(0, FIXED_MAX_RADIUS)  # This explicitly sets polar axis limits
                    
                    # Configure polar plot
                    ax.set_theta_offset(np.pi/2)
                    ax.set_theta_direction(-1)
                    ax.set_thetamin(-180)
                    ax.set_thetamax(180)
                    ax.set_xticks(np.arange(-180, 181, 45) * np.pi/180)
                    ax.set_xticklabels(['-180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°', '180°'])
                    ax.grid(True, alpha=0.3)
                    
                    # Set title
                    control_count = len(control_data) if not control_data.empty else 0
                    exp_count = len(exp_data) if not exp_data.empty else 0
                    ax.set_title(f'Control vs {exp_group}\nControl (n={control_count}) | {exp_group} (n={exp_count})', 
                            pad=20, fontsize=10)
                    
                    # Add legend
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
                
                # Hide unused subplots
                for i in range(len(shape_groups), len(axes)):
                    if i < len(axes):
                        axes[i].set_visible(False)
                
                plt.tight_layout()
                save_path = os.path.join(self.output_dir, f'polar_directions_combined_{shape}_{resolution}.png')
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                save_path_pdf = os.path.join(self.output_dir, f'polar_directions_combined_{shape}_{resolution}.pdf')
                plt.savefig(save_path_pdf, bbox_inches='tight', dpi=300)
                plt.close()
                print(f"Saved combined plot: {save_path}")
                    
        except Exception as e:
            print(f"Error creating combined polar plots: {e}")
            import traceback
            traceback.print_exc()

    def _plot_single_group(self, ax, group_data, label, color, is_experimental=False, max_radius=27):
        """Plot a single group on a polar axis."""
        try:
            theta_data = group_data['signed_direction_rad'].values
            magnitude_data = group_data['displacement_magnitude'].values

            print(f"  Group in single group: {len(theta_data)} data points, magnitude range: {magnitude_data.min():.2f}-{magnitude_data.max():.2f} μm")
            
            if len(theta_data) == 0:
                print(f"No data to plot for {label}")
                return
            
            # Create bins and calculate statistics
            n_bins = 72
            bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            
            # Clip extreme values for better visualization
            magnitude_data = np.clip(magnitude_data, 0, max_radius)

            mean_magnitudes = []
            lower_ci = []
            upper_ci = []
            
            for j in range(len(bin_edges) - 1):
                bin_mask = (theta_data >= bin_edges[j]) & (theta_data < bin_edges[j + 1])
                if j == len(bin_edges) - 2:
                    bin_mask = bin_mask | (theta_data == bin_edges[j + 1])
                
                bin_magnitudes = magnitude_data[bin_mask]
                
                if len(bin_magnitudes) > 0:
                    mean_magnitudes.append(np.mean(bin_magnitudes))
                    
                    if len(bin_magnitudes) >= 5:
                        n_bootstrap = 1000
                        bootstrap_means = []
                        for _ in range(n_bootstrap):
                            bootstrap_sample = np.random.choice(bin_magnitudes, size=len(bin_magnitudes), replace=True)
                            bootstrap_means.append(np.mean(bootstrap_sample))
                        
                        lower_ci.append(np.percentile(bootstrap_means, 16))
                        upper_ci.append(np.percentile(bootstrap_means, 84))
                    else:
                        std_val = np.std(bin_magnitudes) if len(bin_magnitudes) > 1 else mean_magnitudes[-1] * 0.1
                        lower_ci.append(mean_magnitudes[-1] - std_val)
                        upper_ci.append(mean_magnitudes[-1] + std_val)
                else:
                    mean_magnitudes.append(0)
                    lower_ci.append(0)
                    upper_ci.append(0)
            
            # Apply interpolation
            mean_magnitudes = np.array(mean_magnitudes)
            lower_ci = np.array(lower_ci)
            upper_ci = np.array(upper_ci)
                        
            valid_mask = mean_magnitudes > 0
            if np.sum(valid_mask) >= 3:
                valid_centers = bin_centers[valid_mask]
                valid_means = mean_magnitudes[valid_mask]
                valid_lower = lower_ci[valid_mask]
                valid_upper = upper_ci[valid_mask]
                
                valid_centers_wrap = np.concatenate([valid_centers - 2*np.pi, valid_centers, valid_centers + 2*np.pi])
                valid_means_wrap = np.concatenate([valid_means, valid_means, valid_means])
                valid_lower_wrap = np.concatenate([valid_lower, valid_lower, valid_lower])
                valid_upper_wrap = np.concatenate([valid_upper, valid_upper, valid_upper])
                
                try:
                    mean_interp = interpolate.CubicSpline(valid_centers_wrap, valid_means_wrap)
                    lower_interp = interpolate.CubicSpline(valid_centers_wrap, valid_lower_wrap)
                    upper_interp = interpolate.CubicSpline(valid_centers_wrap, valid_upper_wrap)
                    
                    # Create fine grid for smooth plotting
                    theta_fine = np.linspace(-np.pi, np.pi, 360)
                    mean_fine = mean_interp(theta_fine)
                    lower_fine = lower_interp(theta_fine)
                    upper_fine = upper_interp(theta_fine)
                    
                except:
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
            
            # CRITICAL FIX: Ensure all values are properly clipped
            mean_fine = np.clip(mean_fine, 0, max_radius)
            lower_fine = np.clip(lower_fine, 0, max_radius)
            upper_fine = np.clip(upper_fine, 0, max_radius)
            
            # Plot with appropriate style
            linestyle = '--' if is_experimental else '-'
            linewidth = 2 if is_experimental else 2.5
            
            # Only add to legend if we have data
            if np.any(mean_fine > 0):
                # FIX: Use fill_betweenx with explicit bounds to prevent overflow
                ax.fill_between(theta_fine, lower_fine, upper_fine, 
                            alpha=0.2, color=color, label=f'{label} ± STD')
                
                # ALTERNATIVE FIX: Manual polygon creation with strict clipping
                # This ensures the fill stays within bounds
                polygon_theta = np.concatenate([theta_fine, theta_fine[::-1]])
                polygon_r = np.concatenate([lower_fine, upper_fine[::-1]])
                polygon_r = np.clip(polygon_r, 0, max_radius)  # Double clip for safety
                
                ax.fill(polygon_theta, polygon_r, 
                        alpha=0.2, color=color, label=f'{label} ± STD')
                
                ax.plot(theta_fine, mean_fine, linestyle=linestyle, linewidth=linewidth, 
                    color=color, label=f'{label} Mean')
            
            # Add note if data was clipped
            original_max = group_data['displacement_magnitude'].max()
            if original_max > max_radius:
                clipped_count = np.sum(group_data['displacement_magnitude'] > max_radius)
                print(f"  Note: {clipped_count} extreme values clipped for {label} (max was {original_max:.1f}μm)")
                
        except Exception as e:
            print(f"Error plotting single group {label}: {e}")
            
    def plot_all_features_polar_style(self, resolution) -> None:
        """Plot all features in the polar line plot style."""
        try:
            # Create output directory
            polar_dir = os.path.join(self.output_dir, 'polar_style_analysis')
            if not os.path.exists(polar_dir):
                os.makedirs(polar_dir)
            
            # Get the main polar analysis
            self._plot_polar_direction_analysis(resolution, polar_dir)
            
            # Create temporal features in polar-style line plots
            self._plot_temporal_features_polar_style(polar_dir, resolution)
            
            print("All polar-style plots generated successfully!")
            
        except Exception as e:
            print(f"Error in polar-style plotting: {e}")

    def _plot_temporal_features_polar_style(self, output_dir: str, resolution) -> None:
        """Plot temporal features using ACTUAL polar coordinates with enhanced visualization."""
        try:
            # Convert resolution to actual column name
            if resolution == 'minute':
                velocity_col = 'signed_velocity_um_per_minute'
                velocity_label = 'Velocity (μm/minute)'
                title_suffix = 'per Minute'
                time_col = 'minute_start'
            elif resolution == 'hour':  # Default to hour
                velocity_col = 'signed_velocity_um_per_hour' 
                velocity_label = 'Velocity (μm/hour)'
                title_suffix = 'per Hour'
                time_col = 'hour_start'
            else:
                print(f"Unknown resolution: {resolution}")
                return
            
            # Check if velocity column exists
            if velocity_col not in self.df.columns:
                print(f"Velocity column {velocity_col} not found, using hour data")
                velocity_col = 'signed_velocity_um_per_hour'
                velocity_label = 'Velocity (μm/hour)'
                title_suffix = 'per Hour'    
                
            print(f"Using time column: {time_col}")
            
            # Check if the time column exists in the dataframe
            if time_col not in self.df.columns:
                print(f"Warning: Time column '{time_col}' not found in dataframe")
                print(f"Available columns: {[col for col in self.df.columns if 'time' in col.lower() or 'start' in col.lower()]}")
                return        
            
            print(f"Using time column: {time_col}")
            location_types = sorted(self.df['location_type_detailed'].unique())
            
            print(f"Starting polar plot with {len(location_types)} location types: {location_types}")
            
            # Generate distinct colors with special handling for "control"
            colors = {}
            std_alphas = {}
            
            for i, location_type in enumerate(location_types):
                if 'control' in location_type.lower():
                    # Control gets black with gray shading
                    colors[location_type] = 'black'
                    std_alphas[location_type] = 0.2  # Lighter gray for std
                else:
                    # Other location types get distinct colors from colormap
                    non_control_types = [lt for lt in location_types if 'control' not in lt.lower()]
                    color_cycle = plt.cm.Set3(np.linspace(0, 1, len(non_control_types)))
                    control_count = sum(1 for lt in location_types if 'control' in lt.lower())
                    colors[location_type] = color_cycle[i - control_count] if (i - control_count) < len(color_cycle) else 'gray'
                    std_alphas[location_type] = 0.3  # Normal transparency for std
            
            print(f"Color assignment:")
            for loc_type, color in colors.items():
                print(f"  {loc_type}: {color}")
            
            # CORRECTED features - using the columns from movement dynamics
            features = [
                ('net_displacement', 'Net Displacement (μm)', 'Net Displacement in Polar Coordinates'),
                (velocity_col, velocity_label, 'Velocity in Polar Coordinates'),
                ('length_change', 'Length Change (μm)', 'Neurite Length Change in Polar Coordinates'),
                ('displacement_magnitude', 'Movement Magnitude (μm)', 'Movement Magnitude in Polar Coordinates'),
            ]
            
            # Check which features actually exist
            available_features = [f for f, _, _ in features if f in self.df.columns]
            print(f"Available features: {available_features}")
            
            for feature, ylabel, title in features:
                if feature not in self.df.columns:
                    print(f"Skipping {feature} - not in DataFrame")
                    continue
                    
                print(f"Processing feature: {feature}")
                
                try:
                    # Create ACTUAL polar plot
                    fig = plt.figure(figsize=(14, 10))
                    ax = plt.subplot(111, projection='polar')
                    
                    has_plotted_data = False
                    plotted_location_types = []  # Track which locations we actually plot
                    
                    for location_type in location_types:
                        try:
                            loc_data = self.df[self.df['location_type_detailed'] == location_type]
                            print(f"  Location {location_type}: {len(loc_data)} rows")
                            
                            if not loc_data.empty:
                                # Aggregate by time - get mean and std per time point
                                # FIXED: Handle cases where time_col might not exist for this location
                                if time_col not in loc_data.columns:
                                    print(f"    Warning: {time_col} not found for {location_type}")
                                    continue
                                    
                                time_agg = loc_data.groupby(time_col)[feature].agg(['mean', 'std', 'count']).reset_index()
                                time_agg = time_agg.dropna()  # Remove rows with NaN std
                                
                                print(f"    After aggregation: {len(time_agg)} time points")
                                
                                if len(time_agg) > 1:
                                    # Convert time to angles (0 to 2π) - full experimental duration
                                    max_time = time_agg[time_col].max()
                                    min_time = time_agg[time_col].min()
                                    
                                    if max_time == min_time:
                                        print(f"    Warning: Only one unique time point for {location_type}")
                                        continue
                                    
                                    # Convert to numpy arrays to avoid pandas indexing issues
                                    time_values = time_agg[time_col].values
                                    r_mean = time_agg['mean'].values
                                    r_std = time_agg['std'].values
                                    
                                    # Normalize time to 0-2π range
                                    theta = 2 * np.pi * (time_values - min_time) / (max_time - min_time)
                                    
                                    print(f"    Theta range: {theta.min():.2f} to {theta.max():.2f}")
                                    print(f"    Radius range: {r_mean.min():.2f} to {r_mean.max():.2f}")
                                    
                                    # CLOSE THE CIRCLE: Add the first point at the end to complete the circle
                                    theta_closed = np.append(theta, theta[0])
                                    r_mean_closed = np.append(r_mean, r_mean[0])
                                    r_std_closed = np.append(r_std, r_std[0])
                                    
                                    # Plot standard deviation as shaded area with appropriate color
                                    if len(r_std_closed) > 0 and not all(np.isnan(r_std_closed)):
                                        try:
                                            if 'control' in location_type.lower():
                                                # Control: gray shading
                                                fill_color = 'gray'
                                            else:
                                                # Other types: same color as line but with transparency
                                                fill_color = colors[location_type]
                                            
                                            ax.fill_between(theta_closed, 
                                                        r_mean_closed - r_std_closed, 
                                                        r_mean_closed + r_std_closed,
                                                        color=fill_color, 
                                                        alpha=std_alphas[location_type],
                                                        label=f'{location_type} ± STD')
                                        except Exception as fill_error:
                                            print(f"    Fill between error: {fill_error}")
                                    
                                    # Plot main line with location type color
                                    ax.plot(theta_closed, r_mean_closed, 
                                        color=colors[location_type], 
                                        linewidth=3, 
                                        label=location_type, 
                                        alpha=0.9, 
                                        zorder=4)
                                    
                                    # Plot markers - color negative points differently
                                    for j, (t, r_val) in enumerate(zip(theta, r_mean)):
                                        try:
                                            if feature in ['net_displacement', velocity_col, 'length_change'] and r_val < 0:
                                                # Negative points in red with larger size
                                                ax.scatter(t, r_val, color='red', s=60, 
                                                        alpha=0.9, zorder=5, marker='s')  # square for negative
                                            else:
                                                # Positive points in location type color
                                                ax.scatter(t, r_val, color=colors[location_type],
                                                        s=45, alpha=0.8, zorder=5)
                                        except Exception as scatter_error:
                                            print(f"    Scatter error at point {j}: {scatter_error}")
                                    
                                    has_plotted_data = True
                                    plotted_location_types.append(location_type)
                                    
                        except Exception as location_error:
                            print(f"  Error processing location {location_type}: {location_error}")
                            import traceback
                            traceback.print_exc()
                    
                    if not has_plotted_data:
                        print(f"  No data plotted for {feature}, skipping plot")
                        plt.close()
                        continue
                    
                    # Enhanced plot styling
                    ax.set_title(f'{title}\n(Squares=Retracting, Control=Black)', fontsize=14, pad=20)
                    ax.grid(True, alpha=0.3)
                    
                    # Add a circle at radius=0 for reference
                    try:
                        ax.plot(np.linspace(0, 2*np.pi, 100), np.zeros(100), 
                            'k--', alpha=0.5, linewidth=1, label='Zero Reference')
                    except Exception as ref_error:
                        print(f"  Reference line error: {ref_error}")
                    
                    # Customize radial axis to show both positive and negative clearly
                    try:
                        # Get current limits and set symmetric around zero
                        current_ylim = ax.get_ylim()
                        r_max = max(abs(current_ylim[0]), abs(current_ylim[1]))
                        if r_max > 0:
                            ax.set_ylim(-r_max, r_max)
                            print(f"  Radial limits set to: {-r_max:.2f} to {r_max:.2f}")
                    except Exception as ylim_error:
                        print(f"  Y-lim error: {ylim_error}")
                        # Fallback if get_ylim fails
                        pass
                    
                    # Add direction labels for signed features
                    if feature in ['net_displacement', velocity_col, 'length_change']:
                        try:
                            ax.text(np.pi/2, r_max*0.8, 'Advancing\n(Growth)', 
                                ha='center', va='center', fontsize=10, color='black',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
                            ax.text(3*np.pi/2, r_max*0.8, 'Retracting\n(Shrinking)', 
                                ha='center', va='center', fontsize=10, color='black',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
                        except Exception as label_error:
                            print(f"  Label error: {label_error}")
                    
                    # Improved legend - handle duplicates safely
                    try:
                        handles, labels = ax.get_legend_handles_labels()
                        print(f"  Legend items: {len(handles)} handles, {len(labels)} labels")
                        
                        # Remove duplicate labels while preserving order
                        seen = set()
                        unique_handles = []
                        unique_labels = []
                        for handle, label in zip(handles, labels):
                            if label not in seen:
                                seen.add(label)
                                unique_handles.append(handle)
                                unique_labels.append(label)
                        
                        if unique_handles:  # Only add legend if we have items
                            ax.legend(unique_handles, unique_labels, 
                                    bbox_to_anchor=(1.3, 1), loc='upper left', fontsize=10,
                                    framealpha=0.9)
                            print("  Legend created successfully")
                    except Exception as legend_error:
                        print(f"  Legend error: {legend_error}")
                        # Fallback simple legend
                        try:
                            ax.legend(bbox_to_anchor=(1.3, 1), loc='upper left', fontsize=10)
                        except:
                            pass
                    
                    plt.tight_layout()
                    
                    # Use a generic filename without location_type variable
                    save_path = os.path.join(output_dir, f'true_polar_{feature}_{resolution}.png')
                    plt.savefig(save_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    print(f"  Saved plot: {save_path}")
                    
                    # Print summary statistics
                    print(f"\n--- Polar Analysis: {feature} ---")
                    for location_type in plotted_location_types:
                        loc_data = self.df[self.df['location_type_detailed'] == location_type]
                        if not loc_data.empty and feature in loc_data.columns:
                            values = loc_data[feature]
                            pos_fraction = (values > 0).mean() * 100
                            neg_fraction = (values < 0).mean() * 100
                            print(f"{location_type}: "
                                f"Advancing={pos_fraction:.1f}%, "
                                f"Retracting={neg_fraction:.1f}%, "
                                f"Mean={values.mean():.3f}")
                
                except Exception as feature_error:
                    print(f"Error processing feature {feature}: {feature_error}")
                    import traceback
                    traceback.print_exc()
                    try:
                        plt.close()
                    except:
                        pass
                    
        except Exception as e:
            print(f"MAJOR ERROR in polar-style plotting: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_angle_distribution(self, movement_df: pd.DataFrame, resolution: str) -> None:
        """Plot polar plots showing only angle/direction distribution."""
        try:
            # Define color scheme
            color_scheme = {
                'control': 'darkblue',
                'thin': 'orange',
                'mushroom': 'purple', 
                'stubby': 'green'
            }
            
            # Identify control group
            control_groups = [group for group in movement_df['location_type_detailed'].unique() 
                            if 'control' in group.lower()]
            
            if not control_groups:
                print("No control group found for angle distribution plots")
                return
            
            # Use the first control group found
            control_group = control_groups[0]
            print(f"Using control group: {control_group}")
            
            # Get experimental data (everything that's not control)
            experimental_data = movement_df[~movement_df['location_type_detailed'].isin(control_groups)]
            
            # Get unique shapes from experimental data
            shapes = experimental_data['shape'].unique()
            print(f"Found shapes: {shapes}")
            
            # Create ordered groups by shape and pvalue
            ordered_groups = []
            for shape in shapes:
                # Get all groups for this shape
                shape_groups = experimental_data[experimental_data['shape'] == shape]['location_type_detailed'].unique()
                
                # Extract pvalues and sort them (p4, p10, p30)
                pvalue_groups = []
                for group in shape_groups:
                    match = re.search(r'_p(\d+)', group)
                    if match:
                        pval = int(match.group(1))
                        pvalue_groups.append((pval, group))
                
                # Sort by pvalue and add to ordered groups
                pvalue_groups.sort(key=lambda x: x[0])
                for pval, group in pvalue_groups:
                    ordered_groups.append(group)
            
            # Add control at the beginning
            all_groups = [control_group] + ordered_groups
            print(f"All groups for angle distribution: {all_groups}")
            
            # Create TWO types of angle distribution plots:
            # 1. Individual plots for each group
            # 2. Combined plots with control + experimental groups
            
            # PLOT 1: Individual angle distribution plots for each group
            self._create_individual_angle_plots(movement_df, resolution, all_groups, color_scheme)
            
            # PLOT 2: Combined angle distribution plots - control vs each shape type
            self._create_combined_angle_plots(movement_df, resolution, color_scheme, control_group)
            
            print(f"Saved angle distribution plots for {resolution}")
            
        except Exception as e:
            print(f"Error creating angle distribution plots: {e}")
            import traceback
            traceback.print_exc()

    def _create_individual_angle_plots(self, movement_df: pd.DataFrame, resolution: str, groups: list, color_scheme: dict) -> None:
        """Create individual polar plots showing only angle distribution."""
        # Create figure with appropriate number of subplots
        n_groups = len(groups)
        n_cols = min(3, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        # For angle distribution, we use frequency (0-100%) as the radial axis
        FIXED_MAX_FREQUENCY = 10.0  # Maximum frequency percentage
        RADIAL_TICKS = [0, 5, 10]  # Frequency ticks in percentage
        
        fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
        
        for i, group in enumerate(groups, 1):
            if i > n_rows * n_cols:
                break
                    
            ax = fig.add_subplot(n_rows, n_cols, i, projection='polar')
            
            # Get group data
            group_data = movement_df[movement_df['location_type_detailed'] == group]
            
            print(f"📊 Plotting angle distribution for: {group} - Found {len(group_data)} rows")
            
            # Determine color and label
            if 'control' in group.lower():
                group_label = 'Control'
                color = color_scheme['control']
            else:
                group_label = group
                shape = group.split('_')[0]
                color = color_scheme.get(shape, 'gray')
            
            if not group_data.empty:
                theta_data = group_data['signed_direction_rad'].values
                
                print(f"  Group {group}: {len(theta_data)} angle measurements")
                
                # Create bins for angle distribution
                n_bins = 36  # Fewer bins for cleaner angle distribution
                bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                
                # Calculate frequency distribution
                frequencies = []
                for j in range(len(bin_edges) - 1):
                    bin_mask = (theta_data >= bin_edges[j]) & (theta_data < bin_edges[j + 1])
                    if j == len(bin_edges) - 2:
                        bin_mask = bin_mask | (theta_data == bin_edges[j + 1])
                    
                    bin_count = np.sum(bin_mask)
                    frequency = (bin_count / len(theta_data)) * 100  # Convert to percentage
                    frequencies.append(frequency)
                
                frequencies = np.array(frequencies)
                
                # Create circular plot data
                theta_plot = np.concatenate([bin_centers, [bin_centers[0]]])
                freq_plot = np.concatenate([frequencies, [frequencies[0]]])
                
                # Plot the angle distribution as a filled polygon
                ax.fill(theta_plot, freq_plot, alpha=0.3, color=color, label=f'{group_label} Distribution')
                ax.plot(theta_plot, freq_plot, '-', linewidth=2, color=color)
                
                # Configure polar plot
                ax.set_theta_offset(np.pi/2)
                ax.set_theta_direction(-1)
                ax.set_thetamin(-180)
                ax.set_thetamax(180)
                ax.set_xticks(np.arange(-180, 181, 45) * np.pi/180)
                ax.set_xticklabels(['-180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°', '180°'])
                
                # Set radial limits for frequency
                ax.set_ylim(0, FIXED_MAX_FREQUENCY)
                ax.set_yticks(RADIAL_TICKS)
                ax.set_yticklabels([f'{tick}%' for tick in RADIAL_TICKS])
                
                # Add quadrant lines
                quadrant_angles = [-np.pi, -np.pi/2, 0, np.pi/2]
                quadrant_colors = ['red', 'purple', 'green', 'blue']
                
                for angle, qcolor in zip(quadrant_angles, quadrant_colors):
                    ax.plot([angle, angle], [0, FIXED_MAX_FREQUENCY * 0.8], 
                        color=qcolor, linestyle='--', alpha=0.3, linewidth=1)
                
                ax.set_title(f'{group_label}\nAngle Distribution\n(n={len(group_data)} movements)', pad=20, fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Add data quality info
                total_movements = len(group_data)
                ax.text(0.5, -0.15, f'Total: {total_movements} movements', 
                    transform=ax.transAxes, ha='center', fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                if i == 1:
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            else:
                ax.set_ylim(0, FIXED_MAX_FREQUENCY)
                ax.set_yticks(RADIAL_TICKS)
                ax.set_title(f'{group_label}\n(No data)', pad=20, fontsize=12)
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                    ha='center', va='center', fontsize=12, color='red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'angle_distribution_individual_{resolution}.png'),
                bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(self.output_dir, f'angle_distribution_individual_{resolution}.pdf'),
                bbox_inches='tight', dpi=300)
        plt.close()

    def _create_combined_angle_plots(self, movement_df: pd.DataFrame, resolution: str, color_scheme: dict, control_group) -> None:
        """Create combined polar plots showing angle distribution for control vs experimental groups."""
        try:
            # Get control data
            control_data = movement_df[movement_df['location_type_detailed'] == control_group]
            if control_data.empty:
                print(f"No control data found for {control_group}")
                return
            
            # Group experimental data by shape
            experimental_data = movement_df[~movement_df['location_type_detailed'].isin([control_group])]
            shapes = experimental_data['shape'].unique()
            
            print(f"Creating combined angle distribution plots for shapes: {shapes}")
            
            # Create one combined plot per shape type
            for shape in shapes:
                # Get all groups for this shape
                shape_groups = experimental_data[experimental_data['shape'] == shape]['location_type_detailed'].unique()
                shape_groups = [group for group in shape_groups if shape in group]
                
                if not shape_groups:
                    print(f"No groups found for shape: {shape}")
                    continue
                    
                print(f"Processing angle distribution for shape {shape}: {shape_groups}")
                
                # Sort groups by pvalue
                def extract_pvalue(group_name):
                    match = re.search(r'_p(\d+)', group_name)
                    return int(match.group(1)) if match else 0
                
                shape_groups.sort(key=extract_pvalue)
                
                # Create figure for this shape
                n_plots = len(shape_groups)
                n_cols = min(2, n_plots)
                n_rows = (n_plots + n_cols - 1) // n_cols
                
                # For angle distribution, we use frequency (0-100%) as the radial axis
                FIXED_MAX_FREQUENCY =10.0
                RADIAL_TICKS = [0, 5, 10]#, 15, 20, 25]
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), 
                                    subplot_kw=dict(projection='polar'))
                
                # Handle single subplot case
                if n_plots == 1:
                    axes = [axes] if not hasattr(axes, '__len__') else axes
                else:
                    axes = axes.flatten()
                
                # Plot control vs each experimental group
                for i, (ax, exp_group) in enumerate(zip(axes, shape_groups)):
                    if i >= len(axes):
                        break
                        
                    # Get experimental group data
                    exp_data = movement_df[movement_df['location_type_detailed'] == exp_group]
                    
                    # Get colors
                    base_shape = exp_group.split('_')[0]
                    exp_color = color_scheme.get(base_shape, 'gray')
                    
                    print(f"  Plotting angle distribution: {exp_group}")
                    
                    # Clear the axis
                    ax.clear()
                    
                    # Plot control angle distribution
                    if not control_data.empty:
                        self._plot_single_angle_distribution(ax, control_data, 'Control', color_scheme['control'])
                    
                    # Plot experimental group angle distribution
                    if not exp_data.empty:
                        self._plot_single_angle_distribution(ax, exp_data, exp_group, exp_color)
                    
                    # Configure polar plot
                    ax.set_theta_offset(np.pi/2)
                    ax.set_theta_direction(-1)
                    ax.set_thetamin(-180)
                    ax.set_thetamax(180)
                    ax.set_xticks(np.arange(-180, 181, 45) * np.pi/180)
                    ax.set_xticklabels(['-180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°', '180°'])
                    ax.set_ylim(0, FIXED_MAX_FREQUENCY)
                    ax.set_yticks(RADIAL_TICKS)
                    ax.set_yticklabels([f'{tick}%' for tick in RADIAL_TICKS])
                    ax.grid(True, alpha=0.3)
                    
                    # Set title
                    control_count = len(control_data) if not control_data.empty else 0
                    exp_count = len(exp_data) if not exp_data.empty else 0
                    ax.set_title(f'Angle Distribution\nControl vs {exp_group}', 
                            pad=20, fontsize=10)
                    
                    # Add legend
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
                
                # Hide unused subplots
                for i in range(len(shape_groups), len(axes)):
                    if i < len(axes):
                        axes[i].set_visible(False)
                
                plt.tight_layout()
                save_path = os.path.join(self.output_dir, f'angle_distribution_combined_{shape}_{resolution}.png')
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                save_path_pdf = os.path.join(self.output_dir, f'angle_distribution_combined_{shape}_{resolution}.pdf')
                plt.savefig(save_path_pdf, bbox_inches='tight', dpi=300)

                plt.close()
                print(f"Saved combined angle distribution plot: {save_path}")
                    
        except Exception as e:
            print(f"Error creating combined angle distribution plots: {e}")
            import traceback
            traceback.print_exc()

    def _plot_single_angle_distribution(self, ax, group_data, label, color):
        """Plot a single group's angle distribution on a polar axis."""
        try:
            theta_data = group_data['signed_direction_rad'].values
            
            if len(theta_data) == 0:
                return
            
            # Create bins for angle distribution
            n_bins = 36
            bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            
            # Calculate frequency distribution
            frequencies = []
            for j in range(len(bin_edges) - 1):
                bin_mask = (theta_data >= bin_edges[j]) & (theta_data < bin_edges[j + 1])
                if j == len(bin_edges) - 2:
                    bin_mask = bin_mask | (theta_data == bin_edges[j + 1])
                
                bin_count = np.sum(bin_mask)
                frequency = (bin_count / len(theta_data)) * 100  # Convert to percentage
                frequencies.append(frequency)
            
            frequencies = np.array(frequencies)
            
            # Create circular plot data
            theta_plot = np.concatenate([bin_centers, [bin_centers[0]]])
            freq_plot = np.concatenate([frequencies, [frequencies[0]]])
            
            # Plot the angle distribution
            ax.fill(theta_plot, freq_plot, alpha=0.3, color=color, label=f'{label}')
            ax.plot(theta_plot, freq_plot, '-', linewidth=2, color=color)
                
        except Exception as e:
            print(f"Error plotting angle distribution for {label}: {e}")        
            
    def generate_all_plots(self) -> None:
        """Generate all enhanced movement dynamics plots."""
        print("\n" + "="*50)
        print("GENERATING ENHANCED MOVEMENT DYNAMICS PLOTS")
        print("="*50)
        
        # NEW: Create polar direction plots for BOTH resolutions
        # Check which time columns exist and generate plots for each
        time_resolutions = []
        
        if 'minute_start' in self.df.columns:
            time_resolutions.append('minute')
            print("Found minute-level data")
        
        if 'hour_start' in self.df.columns:
            time_resolutions.append('hour') 
            print("Found hour-level data")
        
        if not time_resolutions:
            print("Warning: No time columns found (minute_start or hour_start)")
            return
        
        # Generate polar plots for each available resolution
        for resolution in time_resolutions:
            print(f"\nGenerating polar direction plots for {resolution}-level data...")
            self._plot_polar_directions(self.df, resolution)
            self.plot_all_features_polar_style(resolution)
            self.plot_microns_displacement_analysis(resolution)
            self.plot_enhanced_velocity_analysis(self.df, resolution)
            self._plot_angle_distribution(self.df, resolution)  # NEW: Angle distribution plots

        print(f"\nAll plots saved to: {self.output_dir}")
        print("\nGenerated plots:")
        for file in os.listdir(self.output_dir):
            if file.endswith('.png'):
                file_path = os.path.join(self.output_dir, file)
                file_size = os.path.getsize(file_path) / 1024
                print(f"  - {file} ({file_size:.1f} KB)")
                
def main_single_file():
    """Alternative main function to process a single specific movement dynamics file."""
    try:
        # Define your specific files
        data_paths = [
            "data/results_per_csv/mushroom_p10/movement_dynamics_minute.csv",
            "data/results_per_csv/mushroom_p4/movement_dynamics_minute.csv", 
            "data/results_per_csv/thin_p4_p10/movement_dynamics_minute.csv",
            "data/combined_neurite_analysis/movement_dynamics_complete.csv"
        ]
        
        # Select which file to process (0, 1, or 2)
        selected_index = 3  # Change this index to process different files
        
        data_path = data_paths[selected_index]
        
        if not os.path.exists(data_path):
            print(f"File not found: {data_path}")
            return
        
        # Extract dataset name
        path_parts = Path(data_path).parts
        dataset_name = path_parts[-2]
        
        output_dir = f"data/results_per_csv/{dataset_name}/enhanced_movement_plots"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing: {data_path}")
        print(f"Output directory: {output_dir}")
        
        # Initialize plotter
        plotter = MovementDataPlotter(data_path, output_dir)
        plotter.generate_all_plots()
        
        print(f"✓ Enhanced movement plotting completed for {dataset_name}")
        
    except Exception as e:
        print(f"Error in single file movement plotting: {e}")

if __name__ == "__main__":
       main_single_file()
    
    