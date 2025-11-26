import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
from skimage import morphology, measure
import os 
import pandas as pd
from matplotlib.patches import Circle, Rectangle

def process_fluorescence_channel(image_channel, cell_mask, background_mask):
    """
    Calculates normalized fluorescence intensity by comparing cell and background regions.
    
    This function computes the fluorescence intensity normalized by cell area by 
    comparing mean intensities of cell regions versus background regions.
    
    Parameters:
    -----------
    image_channel : numpy.ndarray
        Single channel fluorescence image
    cell_mask : numpy.ndarray
        Binary mask identifying cell regions (255 for cells, 0 for background)
    background_mask : numpy.ndarray
        Binary mask identifying background regions (255 for background, 0 for cells)
    
    Returns:
    --------
    tuple
        (fluorescence_intensity, cell_mean_intensity, background_mean_intensity)
        - fluorescence_intensity: Normalized intensity (cell_mean - background_mean) / cell_area
        - cell_mean_intensity: Average intensity in cell regions
        - background_mean_intensity: Average intensity in background regions
    """
    # Ensure the masks are boolean arrays for indexing
    cell_mask_bool = cell_mask == 255
    background_mask_bool = background_mask == 255

    # Use boolean arrays to index the image_channel
    cell_pixels = image_channel[cell_mask_bool]
    background_pixels = image_channel[background_mask_bool]
    
    cell_mean_intensity = np.mean(cell_pixels) if cell_pixels.size > 0 else 0
    background_mean_intensity = np.mean(background_pixels) if background_pixels.size > 0 else 0

    # Assuming cell area is the count of true pixels in cell_mask
    cell_area = np.sum(cell_mask_bool)
    fluorescence_intensity = (cell_mean_intensity - background_mean_intensity) / cell_area if cell_area > 0 else 0

    return fluorescence_intensity, cell_mean_intensity, background_mean_intensity

def apply_watershed(image):
    """
    Applies watershed algorithm for image segmentation to separate touching objects.
    
    This function performs watershed segmentation on grayscale images using distance 
    transforms and local maxima to identify distinct regions/cells.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (grayscale or color)
    
    Returns:
    --------
    numpy.ndarray
        Label matrix where each segmented region has a unique integer label
    """
    # Check if image is already in grayscale (single channel)
    if len(image.shape) == 2 or image.shape[2] == 1:
        gray = image  # Image is already a single channel
    else:
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Apply thresholding to create a binary mask
    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = np.mean(blurred_image)  # Adjust as needed
    _, binary_mask = cv2.threshold(blurred_image, threshold, 255, cv2.THRESH_BINARY)

    # Use distance transform to create markers
    distance_transform = ndimage.distance_transform_edt(binary_mask)
    local_max_coordinates = peak_local_max(distance_transform, min_distance=8, labels=binary_mask)
    
    # Initialize an empty array for markers with the same shape as binary_mask
    markers = np.zeros_like(binary_mask, dtype=np.int32)

    # Assign unique values to each peak in markers array
    for i, coord in enumerate(local_max_coordinates, start=1):  # Start labeling from 1
        markers[coord[0], coord[1]] = i

    # Now markers is prepared correctly for watershed
    labels = watershed(-distance_transform, markers, mask=binary_mask)

    return labels


def apply_grabcut(image, binary_mask):
    """
    Refines segmentation using GrabCut algorithm for precise boundary detection.
    
    This function uses the GrabCut algorithm to improve binary mask segmentation 
    by incorporating color and texture information for more accurate cell boundaries.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (grayscale or color)
    binary_mask : numpy.ndarray
        Initial binary segmentation mask
    
    Returns:
    --------
    numpy.ndarray
        Segmented image with refined boundaries
    """
    # Convert image to BGR if it's grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Initialize mask for GrabCut
    grabcut_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Set probable and possible foreground/background
    grabcut_mask[binary_mask == 255] = cv2.GC_PR_FGD
    grabcut_mask[binary_mask == 0] = cv2.GC_BGD

    # Identify definite background and foreground
    # You can use dilation and erosion to create a definite area
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(binary_mask, kernel, iterations=3)  # Dilate to get sure background
    sure_fg = cv2.erode(binary_mask, kernel, iterations=5)   # Erode to get sure foreground

    # Set definite background and foreground
    grabcut_mask[sure_bg == 0] = cv2.GC_BGD
    grabcut_mask[sure_fg == 255] = cv2.GC_FGD

    # Initialize models for GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Apply GrabCut with mask initialization
    cv2.grabCut(image, grabcut_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # Prepare the mask to extract the segmented image
    mask2 = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 1).astype('uint8')
    segmented_image = image * mask2[:, :, np.newaxis]

    return segmented_image

def apply_clahe(image):
    """
    Applies Contrast Limited Adaptive Histogram Equalization to enhance image contrast.
    
    This function improves local contrast in images, particularly useful for 
    enhancing details in fluorescence microscopy images.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Grayscale input image
    
    Returns:
    --------
    numpy.ndarray
        Contrast-enhanced image
    """
    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    clahe_image = clahe.apply(image)
    return clahe_image

def apply_clahe_selectively(original, enhanced, mask):
    """
    Applies CLAHE enhancement selectively to specific regions defined by a mask.
    
    This function blends original and CLAHE-enhanced images based on a binary mask,
    applying enhancement only to masked regions while preserving original elsewhere.
    
    Parameters:
    -----------
    original : numpy.ndarray
        Original grayscale image
    enhanced : numpy.ndarray
        CLAHE-enhanced image
    mask : numpy.ndarray
        Binary mask defining regions to enhance (255 = enhance, 0 = keep original)
    
    Returns:
    --------
    numpy.ndarray
        Selectively enhanced image
    """
    # Use the mask to blend the original and enhanced images
    selectively_enhanced = np.where(mask == 255, enhanced, original)
    return selectively_enhanced

folder_paths = ['data/Experiment adhesion 29_01_24_lif og copy']

# Initialize DataFrame to store results
results_df = pd.DataFrame(columns=[
    'Experiment', 'Key Image Name', 'Image Name', 'Channel',
    'Normalized Fluorescence Intensity SOMA', 'Cell Mean Intensity SOMA', 'Background Mean Intensity SOMA',
    'Normalized Fluorescence Intensity NEURITE', 'Cell Mean Intensity NEURITE', 'Background Mean Intensity NEURITE',
    'Normalized Fluorescence Intensity COMPOUND', 'Cell Mean Intensity COMPOUND', 'Background Mean Intensity COMPOUND'
])

for experiment_index, folder_path in enumerate(folder_paths, start=1):
    # Extract a simple name or identifier for the experiment from the folder path
    experiment_name = os.path.basename(folder_path)
    
    # List all files in the current folder_path
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort(key=lambda x: x[0])
    for file_name in files:
        # Extract key portion of the filename
        key_image_name_parts = file_name.split('.png')[0].split('-')
        key_image_name = key_image_name_parts[-1] if len(key_image_name_parts) > 0 else file_name

        # Full path to the image
        image_path = os.path.join(folder_path, file_name)
        print(f"Processing {image_path}...")

        image = cv2.imread(image_path)
        n_channels = image.shape[-1]
        fig, axes = plt.subplots(2, n_channels*4, figsize=(20, 5))  # Adjust the size as needed

        for channel_index in range(image.shape[-1]):  # Assuming the last dimension is channels
            channel_image = image[:, :, channel_index]            
            # If the channel image is not grayscale, convert it to grayscale
            if len(channel_image.shape) == 3:
                channel_image_gray = cv2.cvtColor(channel_image, cv2.COLOR_BGR2GRAY)
            else:
                channel_image_gray = channel_image            
            image_gray = cv2.fastNlMeansDenoising(channel_image_gray, None, 7, 5, 11)
            _, mask = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_clahe = apply_clahe(image_gray)
            image_clahe_selective = apply_clahe_selectively(image_gray, image_clahe, mask)
            image = cv2.cvtColor(image_clahe_selective, cv2.COLOR_GRAY2BGR)
            watershed_labels = apply_watershed(image)  # Modify apply_watershed if needed to work with SLIC output
            binary_mask = np.uint8(watershed_labels > 0) * 255

            # Finally, refine segmentation with GrabCut
            segmented_image = apply_grabcut(image, binary_mask)
            grabcut_mask = np.where((segmented_image[:, :, 0] == 2) | (segmented_image[:, :, 0] == 0), 0, 1).astype('uint8') * 255
            kernel = np.ones((10, 10), np.uint8)
            grabcut_mask = cv2.morphologyEx(grabcut_mask, cv2.MORPH_OPEN, kernel)
            
            # Convert watershed labels to a visual format
            watershed_visual = np.uint8(watershed_labels > 0) * 255

            # Convert GrabCut refinement result to grayscale for visualization
            grabcut_visual = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        
            # Initialize an image to visualize the somas
            overlay_image = segmented_image.copy()

            # Initialize list for storing individual soma masks
            soma_masks = []
            extended_soma_masks = []
            soma_mask = np.zeros_like(image_gray, dtype=np.uint8)

            # Process the GrabCut mask to find soma regions
            contours, hierarchy = cv2.findContours(grabcut_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if 450 <= cv2.contourArea(contour) <= 50000:
                    # Calculate the bounding rectangle for the contour
                    x, y, w, h = cv2.boundingRect(contour)
                    # Optionally, extend the bounding box by a certain margin, e.g., 10 pixels
                    margin = 10
                    x, y, w, h = x - margin, y - margin, w + 2 * margin, h + 2 * margin

                    # Ensure the bounding box does not exceed the image dimensions
                    x, y = max(x, 0), max(y, 0)
                    w, h = min(w, channel_image_gray.shape[1] - x), min(h, channel_image_gray.shape[0] - y)

                    # Extract the region of interest based on the extended bounding box
                    roi_mask = np.zeros_like(channel_image_gray, dtype=np.uint8)
                    cv2.drawContours(roi_mask, [contour], -1, 255, -1)
                    roi_mask = roi_mask[y:y+h, x:x+w]

                    # Store the soma masks and extended masks
                    extended_soma_masks.append((roi_mask, (x, y, w, h)))   # Assuming these are your area limits for a soma
                    
                    # Initialize a mask for this contour
                    single_soma_mask = np.zeros_like(channel_image_gray, dtype=np.uint8)

                    # Calculate shape descriptors to filter out non-soma regions
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter == 0: continue  # Skip if the perimeter is zero
                    circularity = (4 * np.pi * area) / (perimeter ** 2)
                    
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area == 0: continue  # Skip if the hull area is zero
                    solidity = area / hull_area

                    if  0.001 < circularity < 0.85  and solidity > 0.55:
                        # This contour is considered a soma, draw it on the single_soma_mask
                        cv2.drawContours(soma_mask, [contour], -1, 255, -1)  # Fill the contour in the mask
                        cv2.drawContours(single_soma_mask, [contour], -1, 255, -1)
                        
                        # Add the single soma mask to the list of soma masks
                        soma_masks.append(single_soma_mask)

                        # Draw the soma on the overlay image for visualization
                        cv2.drawContours(overlay_image, [contour], -1, (0, 255, 0), 2)

            kernel = np.ones((3, 3), np.uint8)
            dilated_soma_mask = cv2.dilate(soma_mask, kernel, iterations=1)
            neurite_mask = (watershed_labels > 0).astype(np.uint8)
            neurite_mask[soma_mask > 0] = 0
            neurite_skeleton = morphology.skeletonize(neurite_mask, method='lee').astype(np.uint8)

            # Optionally dilate before labeling to connect disjointed parts
            pre_dilated_skeleton = cv2.dilate(neurite_skeleton, np.ones((2, 2), np.uint8), iterations=3)

            # Label the skeleton to analyze connected neurite components
            label_skel = measure.label(pre_dilated_skeleton, connectivity=2)
            props = measure.regionprops(label_skel)
            min_branch_length = 300  # Define minimum branch length

            # Prepare to clean the labels by removing small branches
            cleaned_labels = np.copy(label_skel)  # Copy to maintain original labels during modifications
            for prop in props:
                if prop.area < min_branch_length:
                    # Set small branches to zero in the labeled skeleton
                    cleaned_labels[cleaned_labels == prop.label] = 0

            # Re-create the cleaned skeleton from the labeled skeleton
            cleaned_skeleton = cleaned_labels > 0

            # Optionally perform morphological closing to smooth out the skeleton
            cleaned_skeleton = morphology.binary_closing(cleaned_skeleton)

            # Convert to uint8 for further processing
            cleaned_skeleton_uint8 = (cleaned_skeleton * 255).astype(np.uint8)

            # Final dilation to enhance visibility and connectivity
            dilated_neurite_skeleton = cv2.dilate(cleaned_skeleton_uint8, np.ones((2, 2), np.uint8), iterations=1)

            # Color the soma regions in green
            soma_overlay = np.zeros_like(image)
            soma_overlay[soma_mask > 0] = (0, 255, 0)  # Green color for somas

            # Color the neurites in red
            neurite_overlay = np.zeros_like(image)
            neurite_overlay[dilated_neurite_skeleton > 0] = (0, 0, 255)  # Red color for neurites

            # Overlay masks on the original image
            overlay_image = cv2.addWeighted(image, 1, soma_overlay, 0.5, 0)  # Blend soma mask
            overlay_image = cv2.addWeighted(overlay_image, 1, neurite_overlay, 0.5, 0)  # Blend neurite mask

            # Plotting channel image
            axes[0, channel_index*4].imshow(channel_image, cmap='gray')
            axes[0, channel_index*4].set_title(f'Ch{channel_index} Original')
            axes[0, channel_index*4].axis('off')

            # Plotting watershed segmentation for the channel
            axes[0, channel_index*4 + 1].imshow(grabcut_mask, cmap='gray')
            axes[0, channel_index*4 + 1].set_title(f'Ch{channel_index} Grabcut Mask')
            axes[0, channel_index*4 + 1].axis('off')

            # Plotting GrabCut refinement for the channel
            axes[0, channel_index*4 + 2].imshow(dilated_neurite_skeleton, cmap='gray')
            axes[0, channel_index*4 + 2].set_title(f'Ch{channel_index} Skeleton Mask')
            axes[0, channel_index*4 + 2].axis('off')

            # Plotting Soma and Neurite for the channel
            axes[0, channel_index*4 + 3].imshow(overlay_image, cmap='gray')
            axes[0, channel_index*4 + 3].set_title(f'Ch{channel_index} Soma Neurite Mask')
            axes[0, channel_index*4 + 3].axis('off')
            
            disk_footprint = morphology.disk(8)  # Adjust the radius as needed for your specific case
            dilated_soma_mask = morphology.binary_dilation(soma_mask, disk_footprint)
            neurite_skeleton_without_somas = np.logical_and(dilated_neurite_skeleton, ~dilated_soma_mask)

            axes[1, channel_index*4].imshow(neurite_skeleton_without_somas, cmap='gray')
            axes[1, channel_index*4].set_title('Dilated Soma Mask')
            axes[1, channel_index*4].axis('off')

            background_mask = np.ones_like(image, dtype=np.uint8) * 255
            dilated_soma_mask_2 = dilated_soma_mask * 255
            background_mask[dilated_soma_mask_2 == 255] = 0
            neurite_skeleton_without_somas_2 = neurite_skeleton_without_somas * 255
            background_mask[neurite_skeleton_without_somas_2 == 255] = 0
            compound_mask = np.maximum(dilated_soma_mask_2, neurite_skeleton_without_somas_2) 

            # Process fluorescence channel for compound mask
            fluorescence_intensity_compound, compound_mean_intensity, background_mean_compound = process_fluorescence_channel(image, compound_mask, background_mask)
            fluorescence_intensity_soma, soma_mean_intensity, background_mean_soma = process_fluorescence_channel(image, dilated_soma_mask_2, background_mask)
            fluorescence_intensity_neurites, neurites_mean_intensity, background_mean_neurites = process_fluorescence_channel(image, neurite_skeleton_without_somas_2, background_mask)

            print(f"Fluorescence Intensity (Compound): {fluorescence_intensity_compound}")
            print(f"Fluorescence Intensity Soma: {fluorescence_intensity_soma}")
            print(f"Fluorescence Intensity Neurites: {fluorescence_intensity_neurites}")

            # Label the full neurite skeleton for a comprehensive view
            labeled_full_skeleton, num_labels = ndimage.label(neurite_skeleton_without_somas)
            props_full_skeleton = measure.regionprops(labeled_full_skeleton)
            color_map = plt.get_cmap('nipy_spectral')  # Use a perceptually uniform colormap

            # Create a colored label image
            colors = color_map(np.linspace(0, 1, num_labels + 1))  # Get colors for each label
            colorful_mask = colors[labeled_full_skeleton]  # Apply colors to labels

            # Convert colorful mask to correct format for overlay
            colorful_mask_rgb = (colorful_mask[:, :, :3] * 255).astype(np.uint8)  # Ignore alpha channel and scale to [0,255]

            # Now apply the weighted overlay
            if colorful_mask_rgb.shape != image.shape:
                colorful_mask_rgb = cv2.resize(colorful_mask_rgb, (image.shape[1], image.shape[0]))

            for prop in props_full_skeleton:
                axes[1, channel_index*4 + 1].text(prop.centroid[1], prop.centroid[0], f'{prop.major_axis_length:.1f}', color='red', fontsize=8, ha='center', va='center')

            # Redisplay the soma regions with annotations on the full skeleton image
            regions_soma = measure.regionprops(measure.label(dilated_soma_mask))
            for i, region in enumerate(regions_soma):
                axes[1, channel_index*4 + 1].text(region.centroid[1], region.centroid[0], f'Soma {i+1}', color='yellow', fontsize=10, ha='center', va='center')

            axes[1, channel_index*4 + 1].imshow(neurite_skeleton_without_somas, cmap='gray')
            axes[1, channel_index*4 + 1].set_title('Neurite and  Soma Mask')
            axes[1, channel_index*4 + 1].axis('off')

            disk_footprint = morphology.disk(10)  # You might adjust this based on visual inspection
            dilated_soma_mask_association = morphology.binary_dilation(soma_mask, disk_footprint)
            labeled_soma_mask, _ = ndimage.label(dilated_soma_mask_association)
            soma_regions = measure.regionprops(labeled_soma_mask)
            neurite_regions = measure.regionprops(labeled_full_skeleton, intensity_image=labeled_soma_mask)

            neurite_to_soma = {}

            for neurite in neurite_regions:
                # Ensure there are entries in the intensity image to analyze
                if neurite.intensity_image.size > 0 and np.bincount(neurite.intensity_image.astype(int).flat).size > 1:
                    soma_label = np.bincount(neurite.intensity_image.astype(int).flat)[1:].argmax() + 1  # Correctly offset by ignoring background
                    if soma_label > 0:  # Check that it's a valid label, not background
                        neurite_to_soma[neurite.label] = soma_label
                else:
                    # Handle cases where no valid soma association is found
                    neurite_to_soma[neurite.label] = None

            print("Neurite to Soma Mapping:", neurite_to_soma)

            ax = axes[1, channel_index*4 + 2]
            ax.imshow(image, cmap='gray')  # Display the original image in the background
            ax.set_title('Neurite to Soma Associations')
            ax.axis('off')
            # Plotting the soma regions with one color
            for soma in soma_regions:
                minr, minc, maxr, maxc = soma.bbox
                rect = Rectangle((minc, minr), maxc - minc, maxr - minr, edgecolor='red', facecolor='none', lw=2)
                ax.add_patch(rect)
                ax.text(minc + (maxc - minc) / 2, minr + (maxr - minr) / 2, f'Soma {soma.label}',
                        color='white', ha='center', va='center', fontsize=8)

            # Plotting the neurite regions with another color
            for neurite in neurite_regions:
                minr, minc, maxr, maxc = neurite.bbox
                rect = Rectangle((minc, minr), maxc - minc, maxr - minr, edgecolor='blue', facecolor='none', lw=2)
                ax.add_patch(rect)
                if neurite.label in neurite_to_soma and neurite_to_soma[neurite.label] is not None:
                    associated_soma = neurite_to_soma[neurite.label]
                    if associated_soma <= len(soma_regions):  # Additional check for safety
                        soma_centroid = soma_regions[associated_soma - 1].centroid  # -1 because labels are 1-indexed
                        neurite_centroid = (neurite.centroid[1], neurite.centroid[0])  # Switch x and y for proper plotting
                        ax.plot([neurite_centroid[0], soma_centroid[1]], [neurite_centroid[1], soma_centroid[0]], 'yellow')
                        ax.text(neurite_centroid[0], neurite_centroid[1], f'{neurite.label}->Soma {associated_soma}',
                                color='lightgreen', ha='center', va='center', fontsize=6)

            # Create a mask for coloring
            height, width = image.shape[:2]
            colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

            # Define a colormap
            color_map = plt.get_cmap('nipy_spectral')
            # Find the max label for color mapping, ensuring that None values are ignored
            num_labels = max((soma for soma in neurite_to_soma.values() if soma is not None), default=0) + 1

            # Color the somas
            soma_regions = measure.regionprops(labeled_soma_mask)
            for soma in soma_regions:
                if soma.label in neurite_to_soma.values():  # Check if soma is associated
                    color = color_map(soma.label / num_labels)[:3]  # Get color, discard alpha if exists
                    color = (np.array(color) * 255).astype(np.uint8)  # Convert to 0-255 scale
                    for coord in soma.coords:  # Apply color
                        colored_mask[coord[0], coord[1], :] = color

            # Color the neurites with the same color as their associated soma
            for neurite in neurite_regions:
                if neurite.label in neurite_to_soma:
                    associated_soma = neurite_to_soma[neurite.label]
                    if associated_soma:
                        color = color_map(associated_soma / num_labels)[:3]
                        color = (np.array(color) * 255).astype(np.uint8)
                        for coord in neurite.coords:
                            colored_mask[coord[0], coord[1], :] = color

            # Overlay the colored mask onto the original image
            overlay_image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
            axes[1, channel_index*4 + 3].imshow(overlay_image)
            axes[1, channel_index*4 + 3].set_title('Overlay of Associated Neurites and Somas')
            axes[1, channel_index*4 + 3].axis('off')            
 
            # Append results to the DataFrame
            new_row = {
                'Experiment': experiment_name, #experiment_name,  # or 'Experiment Index': experiment_index if you added that transformation
                'Key Image Name': key_image_name,
                'Image Name': file_name,
                'Channel': channel_index,
                'Normalized Fluorescence Intensity SOMA': fluorescence_intensity_soma,
                'Cell Mean Intensity SOMA': soma_mean_intensity,
                'Background Mean Intensity SOMA': background_mean_soma,
                'Normalized Fluorescence Intensity NEURITE': fluorescence_intensity_neurites,
                'Cell Mean Intensity NEURITE': neurites_mean_intensity,
                'Background Mean Intensity NEURITE': background_mean_neurites,
                'Normalized Fluorescence Intensity COMPOUND': fluorescence_intensity_compound,
                'Cell Mean Intensity COMPOUND': compound_mean_intensity,
                'Background Mean Intensity COMPOUND': background_mean_compound
            }
            results_df = results_df._append(new_row, ignore_index=True)

        plt.tight_layout()
        plt.savefig(f'output_{experiment_name}_{key_image_name}.png', dpi=300)
        plt.close(fig)  # Close the figure to free memory
        # plt.show()
        

results_df.to_csv('soma_neurite_separate_data_first_clean_changed.csv', index=False)
print(results_df.head())
