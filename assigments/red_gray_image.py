import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color

def show_edited_and_original_pic_and_histograms_very_compact(image_path):
    image = io.imread(image_path)
    hsv_image = color.rgb2hsv(image)
    
    lower_red_mask = (hsv_image[:,:,0] >= 0) & (hsv_image[:,:,0] <= 0.07)
    upper_red_mask = (hsv_image[:,:,0] >= 0.90) & (hsv_image[:,:,0] <= 1.0)
    red_mask = lower_red_mask | upper_red_mask
        
    gray_image = color.rgb2gray(image)
    
    # Apply the mask and keep original colors
    final_image = np.zeros_like(image)
    for i in range(3): 
        final_image[:,:,i] = np.where(red_mask, image[:,:,i], gray_image*255)
    
    # Further reduced figure size
    fig, axs = plt.subplots(4, 1, figsize=(6, 12))  # Reduced width for compactness
    
    # Display images
    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(final_image)
    axs[1].set_title('Edited Image')
    axs[1].axis('off')
    
    # Extract hue values from the original and edited images
    original_hue_values = hsv_image[:,:,0].flatten()
    edited_hue_values = hsv_image[:,:,0][red_mask].flatten()
    
    # Plot histograms
    axs[2].hist(original_hue_values, bins=30, color='blue', alpha=0.7)
    axs[2].set_title('Original Image Hue Distribution')
    axs[2].set_xlabel('Hue Value')
    axs[2].set_ylabel('Frequency')
    
    axs[3].hist(edited_hue_values, bins=30, color='red', alpha=0.7)
    axs[3].set_title('Edited Image Hue Distribution (Red Parts)')
    axs[3].set_xlabel('Hue Value')
    axs[3].set_ylabel('Frequency')
    
    plt.tight_layout(pad=1.0)  # Reduced padding between plots
    plt.show()


show_edited_and_original_pic_and_histograms_very_compact('golden-gate-bridge-gettyimages-671734928.jpg')
