#Whitebalance

# Using Gray World Assumption (Manual White Balance)
# The Gray World Assumption method assumes that the average color in an image should be neutral gray, 
# so each channel (R, G, B) is scaled accordingly.

import cv2
import numpy as np

file_name_in = 'DSC03398-2.JPG'
file_name_out = 'DSC03398_wb-2.JPG'
path_in = '/Users/simon/Documents/Master/Masterarbeit/aktuelles Thema/Bilder_Blätter_LAI/Beispiel/' + file_name_in
path_out = '/Users/simon/Documents/Master/Masterarbeit/aktuelles Thema/Bilder_Blätter_LAI/Beispiel/' + file_name_out



def white_balance_grayworld(image):
    # Split the image into its B, G, and R components
    b, g, r = cv2.split(image)
    
    # Compute the average of each channel
    avg_b = np.mean(b)
    avg_g = np.mean(g)
    avg_r = np.mean(r)
    
    # Compute the average intensity across all channels
    avg_gray = (avg_b + avg_g + avg_r) / 3
    
    # Scale each channel so that their average becomes equal to the gray average
    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r
    
    # Apply the scaling to the B, G, R channels
    b = cv2.multiply(b, scale_b)
    g = cv2.multiply(g, scale_g)
    r = cv2.multiply(r, scale_r)
    
    # Merge the channels back into a color image
    white_balanced_image = cv2.merge([b, g, r])
    
    return white_balanced_image

# Example usage
image = cv2.imread(path_in)

# Apply automated white balance
white_balanced_image = white_balance_grayworld(image)

# Save or display the result
cv2.imwrite(path_out , white_balanced_image)