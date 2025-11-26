import cv2
import numpy as np
import matplotlib.pyplot as plt

class ROISelector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.original_height, self.original_width = self.original_image.shape[:2]
        
        # Calculate display size that fits the screen
        self.display_scale = self.calculate_display_scale()
        self.display_width = int(self.original_width * self.display_scale)
        self.display_height = int(self.original_height * self.display_scale)
        
        # Create display image
        self.display_image = cv2.resize(self.original_image, 
                                      (self.display_width, self.display_height))
        
        self.clone = self.display_image.copy()
        self.rois = []  # Will store ROIs in display coordinates
        self.original_rois = []  # Will store ROIs in original coordinates
        self.current_roi = []
        self.drawing = False
        self.roi_count = 0
        self.max_rois = 6
        
    def calculate_display_scale(self):
        # Get screen dimensions (approximate)
        screen_width = 1920  # You can adjust these based on your screen
        screen_height = 1080
        
        # Calculate scale to fit screen (with some margin)
        scale_x = (screen_width - 100) / self.original_width
        scale_y = (screen_height - 200) / self.original_height
        
        # Use the smaller scale to ensure image fits completely
        return min(scale_x, scale_y, 1.0)  # Don't scale up if image is smaller than screen
    
    def display_to_original_coords(self, x, y):
        """Convert display coordinates back to original image coordinates"""
        orig_x = int(x / self.display_scale)
        orig_y = int(y / self.display_scale)
        # Ensure coordinates are within image bounds
        orig_x = max(0, min(orig_x, self.original_width - 1))
        orig_y = max(0, min(orig_y, self.original_height - 1))
        return orig_x, orig_y
    
    def original_to_display_coords(self, x, y):
        """Convert original coordinates to display coordinates"""
        disp_x = int(x * self.display_scale)
        disp_y = int(y * self.display_scale)
        return disp_x, disp_y

    def draw_roi(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.roi_count < self.max_rois:
                self.drawing = True
                self.current_roi = [(x, y)]
                
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                temp_image = self.clone.copy()
                cv2.rectangle(temp_image, self.current_roi[0], (x, y), (0, 255, 0), 2)
                cv2.imshow('Select 6 ROIs - Press q when done', temp_image)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.roi_count < self.max_rois:
                self.drawing = False
                self.current_roi.append((x, y))
                
                # Store display coordinates for drawing
                self.rois.append(self.current_roi.copy())
                
                # Convert to original coordinates and store
                orig_start = self.display_to_original_coords(self.current_roi[0][0], self.current_roi[0][1])
                orig_end = self.display_to_original_coords(self.current_roi[1][0], self.current_roi[1][1])
                self.original_rois.append([orig_start, orig_end])
                
                self.roi_count += 1
                
                # Draw the final ROI on the clone
                cv2.rectangle(self.clone, self.current_roi[0], self.current_roi[1], (0, 255, 0), 2)
                
                # Add ROI number
                x1, y1 = self.current_roi[0]
                cv2.putText(self.clone, f'ROI {self.roi_count}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                cv2.imshow('Select 6 ROIs - Press q when done', self.clone)
                
                # Print coordinates in both display and original scale
                print(f"ROI {self.roi_count} selected:")
                print(f"  Display coords: {self.current_roi[0]} to {self.current_roi[1]}")
                print(f"  Original coords: {orig_start} to {orig_end}")
                
                if self.roi_count == self.max_rois:
                    print("All 6 ROIs selected. Press 'q' to generate mask or 'r' to reset.")

    def select_rois(self):
        window_name = 'Select 6 ROIs - Press q when done'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.display_width, self.display_height)
        cv2.setMouseCallback(window_name, self.draw_roi)
        
        print(f"Image display info:")
        print(f"  Original size: {self.original_width} x {self.original_height}")
        print(f"  Display size: {self.display_width} x {self.display_height}")
        print(f"  Scale factor: {self.display_scale:.3f}")
        print("\nInstructions:")
        print("1. Click and drag to draw rectangles for 6 ROIs")
        print("2. Press 'r' to reset all ROIs")
        print("3. Press 'q' to generate mask when done")
        print("4. Press 'c' to cancel")
        
        while True:
            cv2.imshow(window_name, self.clone)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') and self.roi_count == self.max_rois:
                break
            elif key == ord('r'):
                self.reset_rois()
            elif key == ord('c'):
                cv2.destroyAllWindows()
                return None
                
        cv2.destroyAllWindows()
        return self.original_rois
    
    def reset_rois(self):
        self.rois = []
        self.original_rois = []
        self.current_roi = []
        self.roi_count = 0
        self.clone = self.display_image.copy()
        print("All ROIs reset. You can draw again.")
    
    def generate_mask(self):
        if len(self.original_rois) != self.max_rois:
            print(f"Error: Expected {self.max_rois} ROIs, but got {len(self.original_rois)}")
            return None
        
        # Create an empty mask with the original image dimensions
        mask = np.zeros((self.original_height, self.original_width), dtype=np.uint8)
        
        # Fill each ROI in the mask using original coordinates
        for i, roi in enumerate(self.original_rois):
            x1, y1 = roi[0]
            x2, y2 = roi[1]
            
            # Ensure coordinates are in the correct order
            x_start, x_end = min(x1, x2), max(x1, x2)
            y_start, y_end = min(y1, y2), max(y1, y2)
            
            # Set the ROI area to white (255) in the mask
            mask[y_start:y_end, x_start:x_end] = 255
        
        return mask
    
    def create_display_image_with_rois(self):
        """Create a version of the display image with all ROIs drawn"""
        display_with_rois = self.display_image.copy()
        
        for i, roi in enumerate(self.rois):
            # Draw ROI rectangle
            cv2.rectangle(display_with_rois, roi[0], roi[1], (0, 255, 0), 2)
            
            # Add ROI number
            x1, y1 = roi[0]
            cv2.putText(display_with_rois, f'ROI {i+1}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return display_with_rois
    
    def visualize_results(self, mask):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(self.original_image)
        axes[0].set_title(f'Original Image\n{self.original_width} x {self.original_height}')
        axes[0].axis('off')
        
        # Display image with ROIs
        display_with_rois = self.create_display_image_with_rois()
        axes[1].imshow(display_with_rois)
        axes[1].set_title(f'Display with ROIs\n{self.display_width} x {self.display_height}')
        axes[1].axis('off')
        
        # Mask (in original size)
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title(f'Combined Mask\n{self.original_width} x {self.original_height}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    # Specify your image path
    image_path = "data/outputs_by_location/stubby_p4/pngs/frame_0001.png"
    
    try:
        # Initialize ROI selector
        selector = ROISelector(image_path)
        
        # Select ROIs
        rois = selector.select_rois()
        
        if rois is not None:
            # Generate mask
            mask = selector.generate_mask()
            
            if mask is not None:
                # Visualize results
                selector.visualize_results(mask)
                
                # Save the mask
                mask_path = "combined_roi_mask.png"
                cv2.imwrite(mask_path, mask)
                print(f"\nMask saved as: {mask_path}")
                print(f"Mask size: {mask.shape[1]} x {mask.shape[0]} (original image size)")
                
                # Print ROI coordinates
                print("\nROI Coordinates (Original Image Scale):")
                for i, roi in enumerate(rois, 1):
                    print(f"ROI {i}: Top-left {roi[0]}, Bottom-right {roi[1]}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
