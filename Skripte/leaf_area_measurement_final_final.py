# dependencies
import cv2
import numpy as np
import os
import csv

# function to warp image with aruco markers
def warp_image(image, width_mm, height_mm):
    # load ArUco markers dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # detect ArUco markers in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and len(ids) >= 4:
        # calculate centroids of little squares
        quadrilateral_centroids = []
        for quadrilateral in corners:
            points = quadrilateral[0]
            centroid = np.mean(points, axis=0)
            quadrilateral_centroids.append(centroid)
        quadrilateral_centroids = np.array(quadrilateral_centroids)

        # calculate overall centroid
        overall_centroid = np.mean(quadrilateral_centroids, axis=0)

        # detect corners of large rectangle
        large_quad_corners = []
        
        for i, quadrilateral in enumerate(corners):
            points = quadrilateral[0]
            q_centroid = quadrilateral_centroids[i]
            direction_vector = q_centroid - overall_centroid
            direction_vector /= np.linalg.norm(direction_vector)  # normalisation
            # project points on direction vector
            vectors = points - overall_centroid
            projections = np.dot(vectors, direction_vector)
            max_index = np.argmax(projections)
            #min_index = np.argmin(projections)
            large_quad_corners.append(points[max_index])

        large_quad_corners = np.array(large_quad_corners)

        sorted_indices = np.argsort(ids, axis=None)
        pts_src = large_quad_corners[sorted_indices]

        # define target points for user-specific size (e.g., DIN A4: 210 mm x 297 mm) (test: 197 mm x 286 mm)
        dpi = 72
        width, height = round(width_mm * dpi/25.4), round(height_mm * dpi/25.4) 
        pts_dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        # perform perspective transformation
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(image, M, (width, height))
        
        print('Warped successfully')
        return(warped)
    else:
        print("Not all ArUco markers detected.")
        print(len(corners))
        return(None)

# function to adapt white balance using Gray World Assumption (Manual White Balance)
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

    print('White-balanced successfully')
    return white_balanced_image

def draw_white_corners(image, corner_square_length_mm, width_image_mm, height_image_mm):
    # Image dimensions in pixels
    height_pixels, width_pixels = image.shape[:2]
    
    # Convert the square length from millimeters to pixels
    square_length_pixels_w = int(np.ceil((corner_square_length_mm / width_image_mm) * width_pixels))
    square_length_pixels_h = int(np.ceil((corner_square_length_mm / height_image_mm) * height_pixels))
    
    # Ensure that largest square is taken if difference in dimensions due to rounding
    square_length_pixels = max(square_length_pixels_w, square_length_pixels_h)
    
    # Fill the four corners of the image with white
    # Top-left corner
    image[0:square_length_pixels, 0:square_length_pixels] = (255, 255, 255)
    # Top-right corner
    image[0:square_length_pixels, -square_length_pixels:] = (255, 255, 255)
    # Bottom-left corner
    image[-square_length_pixels:, 0:square_length_pixels] = (255, 255, 255)
    # Bottom-right corner
    image[-square_length_pixels:, -square_length_pixels:] = (255, 255, 255)
    
    print('Whitened markers successfully')

    return image

def mask_background_with_edges(image, lower_bound, upper_bound):
    # Step 1: Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 2: Create a mask for the background based on HSV thresholds
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Step 3: Invert mask for object isolation
    mask_inv = cv2.bitwise_not(mask)

    # Step 4: Refine the mask with morphological operations (optional)
    kernel = np.ones((3, 3), np.uint8)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)

    # Step 5: Apply the mask to isolate the objects (leaves)
    masked = cv2.bitwise_and(image, image, mask=mask_inv)

    # Step 6: Apply edge detection to further refine the leaf boundaries
    edges = apply_edge_detection(image)

    # Step 7: Combine the edge-detected image with the HSV mask
    combined_mask = cv2.bitwise_and(masked, masked, mask=edges)

    return combined_mask


def apply_edge_detection(image):
    """
    Use Canny edge detection to detect the boundaries of the leaves.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 100, 200)  # You can adjust these values for better results

    # Invert the edges to create a mask for leaf areas
    edges_inv = cv2.bitwise_not(edges)

    # Use morphological closing to fill small gaps in edges
    kernel = np.ones((3, 3), np.uint8)
    edges_inv = cv2.morphologyEx(edges_inv, cv2.MORPH_CLOSE, kernel)

    return edges_inv

def calculate_leaf_area_in_images(folder_path, total_area_mm2, output_csv):
    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))]
    
    # Open the CSV file for writing
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Filename', 'leaf area mm^2'])
        
        print(image_files)

        for image_file in image_files:
            # Read the image
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Define a threshold to classify black pixels (value 0)
            _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
            
            # Count the number of non-black pixels (leaf area)
            non_black_pixels = np.sum(binary_image == 255)
            
            # Calculate the total number of pixels in the image
            total_pixels = image.shape[0] * image.shape[1]
            
            # Calculate the non-black (leaf) area in mm² (proportional to the total area)
            leaf_area_mm2 = (non_black_pixels / total_pixels) * total_area_mm2
            
            # Write the result to the CSV file
            writer.writerow([image_file, f"{leaf_area_mm2:.2f}"])


### USER INPUT
path_dir_in = '/Users/simon/Documents/Master/Masterarbeit/aktuelles Thema/Bilder_Blätter_LAI/LitterTraps_240930/'
path_dir_out = '/Users/simon/Documents/Master/Masterarbeit/aktuelles Thema/Bilder_Blätter_LAI/LitterTraps_240930_processed/'

# horizontal and vertical distances between ArUco markers (mm, outer corners)
# algorithm sorts corner ids (ascending) and transforms images that lowest id is at upper-left corner. No manual image rotations necessary 
width_mm = 980
height_mm = 1980

# color boundaries for the background (white/grey) in HSV color space
lower_bound = np.array([0, 0, 110]) #130
upper_bound = np.array([190, 45, 255]) #+10

files_in = os.listdir(path_dir_in)

for file_in in files_in:
    if file_in.split('.')[-1] in ['jpg','JPG','png','PNG','jpeg','JPEG']:
        print(file_in)

        # read image
        image = cv2.imread(os.path.join(path_dir_in, file_in))

        # warp image
        image_processed = warp_image(image, width_mm, height_mm)

        # white balance image
        image_processed = white_balance_grayworld(image_processed)

        # whiten aruco markers
        image_processed = draw_white_corners(image_processed,
                                             corner_square_length_mm=45,
                                             width_image_mm=width_mm,
                                             height_image_mm=height_mm)

        # mask background
        image_processed = mask_background_with_edges(image_processed, lower_bound=lower_bound, upper_bound=upper_bound)

        # write image
        file_out_name, file_out_ext = file_in.split('.')
        cv2.imwrite(f"{os.path.join(path_dir_out,file_out_name)}_processed.{file_out_ext}", image_processed)

calculate_leaf_area_in_images(folder_path=path_dir_out,
                              total_area_mm2=width_mm*height_mm, 
                              output_csv= os.path.join(path_dir_out, 'leaf_area.csv'))