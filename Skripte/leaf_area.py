import cv2
import numpy as np

#referennz 3.8 * 5.7 cm
referenz_area = 6*(3.8*5.7)

# set paths
file_name_in = 'DSC03398_warped-2.JPG'
file_name_out = 'DSC03398_warped_segmented-2.JPG'
path_in = '/Users/simon/Documents/Master/Masterarbeit/aktuelles Thema/Bilder_Blätter_LAI/Beispiel/' + file_name_in
path_out = '/Users/simon/Documents/Master/Masterarbeit/aktuelles Thema/Bilder_Blätter_LAI/Beispiel/' + file_name_out

def whiten_aruco_markers(image, detector): # First Try not working yet
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the image
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        # Iterate over detected markers
        for marker_corners in corners:
            # Convert the corners to integer pixel positions
            pts = marker_corners[0].astype(np.int32)

            # Fill the marker region with white color
            cv2.fillPoly(image, [pts], (255, 255, 255))
    
    return image

#Get the image dimensions to calculate the area
def get_image_dimensions(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Get the dimensions of the image
    height, width, channels = image.shape
    
    print(f"Image Dimensions: {width}x{height} pixels")
    print(f"Number of Channels: {channels}")
    
    return width, height, channels

# Example usage
width, height, channels = get_image_dimensions(path_in)

distance_in_pixels = height
distance_in_mm = 1905 #real distance of the Aruco markers in mm

def segment_objects(image_path, write_file=True): #segmentation and calculation of the area and error
    # load image
    image = cv2.imread(image_path)
    
    # convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # opencv internally uses BGR instead of RGB
    
    # define thresholds for background (white/grey)
    lower_bound = np.array([0, 0, 180]) 
    upper_bound = np.array([180, 50, 255])
    
    # create mask for background
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # invert mask for object isolation
    mask_inv = cv2.bitwise_not(mask)
    
    # refine mask with morphologic operations
    kernel = np.ones((3,3),np.uint8)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)
    
    # appy mask on original image
    segmented = cv2.bitwise_and(image, image, mask=mask_inv)
    
    if write_file:
        # write segemented image
        cv2.imwrite(path_out, segmented)

     # Calculate the area of the mask in pixels
    mask_area_in_pixels = np.count_nonzero(mask_inv)

    # Calculate the pixel-to-mm scale factor
    scale_factor = distance_in_mm / distance_in_pixels
    
    # Calculate the area in mm²
    mask_area_in_mm2 = mask_area_in_pixels * (scale_factor ** 2)

    # Convert mm² to cm² (1 cm² = 100 mm²)
    mask_area_in_cm2 = mask_area_in_mm2 / 100
    
    print(f"Mask Area (in pixels): {mask_area_in_pixels} pixels")
    print(f"Mask Area (in mm²): {mask_area_in_mm2} mm²")
    print(f"Mask Area (in cm²): {mask_area_in_cm2} cm²")
    print(f"Referenz area: {referenz_area} cm²")

    #Error: Referenz vs Masked area

    error = referenz_area - mask_area_in_cm2 
    error_percent = (error / referenz_area) * 100

    print(f"Error: {error} cm²")
    print(f"Error (%): {error_percent} %")
    return mask_inv, mask_area_in_pixels, mask_area_in_mm2


# call segmentation
test = segment_objects(path_in)

#np.unique_counts(test)
