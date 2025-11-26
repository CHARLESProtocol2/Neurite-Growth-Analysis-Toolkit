# Neurite Morphology and Dynamics Analysis Toolkit

A comprehensive computational pipeline for analyzing neurite growth dynamics and morphology in response to micro-patterned substrates. This toolkit provides end-to-end analysis from fluorescent image segmentation to sophisticated movement dynamics characterization, specifically designed for studying how neurites interact with different pillar geometries and surface patterns.

The repository contains two main analytical workflows:
1. **Fluorescent Image Analysis**: Quantifies soma and neurite characteristics from multi-channel fluorescence images
2. **Neurite Tracking & Dynamics**: Tracks neurite movements over time and analyzes behavior differences between pillar and control regions

## Project Structure & Workflow

### 1. Fluorescent Image Analysis Pipeline

**`fluorescent_image_analysis.py`**
- **Purpose**: Segments multi-channel fluorescence images to quantify soma and neurite characteristics
- **Input**: PNG images of fluorescence staining
- **Output**: 
  - Debug images showing channel segmentation results
  - CSV file with quantified soma and neurite measurements
- **Key Features**: Multi-channel segmentation, intensity measurement, morphological characterization

**`fluorescent_images_data_analysis.py`**
- **Purpose**: Statistical analysis of fluorescence data based on pillar type, substrate shape, and channel
- **Input**: CSV output from fluorescent_image_analysis.py
- **Output**: Comparative analysis plots and statistics grouped by experimental conditions

### 2. Neurite Tracking & Dynamics Pipeline

**Step 1: Region of Interest Definition**
**`ROI_Selection_Mask_Generator.py`**
- **Purpose**: Dynamically selects quadrants (6 regions) in video frames to create binary masks
- **Function**: Separates neurites on pillars (mask value = 1) from control/flat regions (mask value = 0)
- **Output**: Binary masks for spatial segmentation of neurite environments

**Step 2: Video Preprocessing**
**`preprocessing_helpers.py`**
- **Purpose**: Utility functions for video frame processing and neurite annotation handling
- **Contains**: Helper functions for data normalization, coordinate transformation, and annotation parsing

**`preprocessing_bulk_vids.py`**
- **Purpose**: Batch processes video frames and converts NeuronJ NDF files to analyzable formats
- **Input**: Raw video frames + NeuronJ annotation files
- **Output**: 
  - Processed CSV files for tracking analysis
  - Debug visualization videos
  - Organized folder structure for each video dataset

**Step 3: Movement Analysis**
**`post_processing_data_analysis_singlecsv.py`**
- **Purpose**: Aggregates tracking data from multiple experiments into comprehensive analysis datasets
- **Output**: Multiple CSV dataframes for:
  - Movement dynamics (velocity, displacement, direction)
  - Overall neurite summary statistics
  - Temporal progression patterns

**`post_processing_mov_dynamics.py`**
- **Purpose**: Generates explanatory visualizations comparing neurite behavior on pillars vs. flat control regions
- **Features**: Polar plots of movement directions, velocity distributions, temporal progression analysis
- **Output**: Publication-quality figures demonstrating behavioral differences

## Applications

- **Neurite-Substrate Interactions**: Quantify how neurites respond to micro-patterned surfaces
- **Biomaterial Evaluation**: Test different substrate geometries for neural tissue engineering

## Output Features

- **Spatial Analysis**: Regional comparison of neurite behavior
- **Temporal Dynamics**: Minute-by-minute and hour-level resolution tracking
- **Movement Classification**: Advancement vs. retraction behavior quantification
- **Directional Preference**: Polar analysis of growth cone guidance
- **Morphological Metrics**: Length, straightness, tortuosity, and branching patterns

## Publication-Ready Analysis

This pipeline generates comprehensive datasets and visualizations suitable for direct inclusion in scientific publications, including:

- Statistical comparisons between experimental conditions
- Time-lapse behavior analysis
- Directional preference plots with confidence intervals
- Movement dynamics across multiple temporal scales

---

*For detailed usage instructions and parameter tuning, refer to individual script documentation and example configurations.*
