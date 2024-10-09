#whitening arcu markers

import cv2
import numpy as np

file_name_in = 'DSC03390_warped.JPG'
file_name_out = 'DSC03390_warped_no_markers.JPG'
path_in = '/Users/simon/Documents/Master/Masterarbeit/aktuelles Thema/Bilder_Blätter_LAI/Beispiel/' + file_name_in
path_out = '/Users/simon/Documents/Master/Masterarbeit/aktuelles Thema/Bilder_Blätter_LAI/Beispiel/' + file_name_out


def whiten_aruco_markers(image, detector):
    """
    Detect and whiten ArUco markers in the provided image.
    
    Args:
        image (numpy.ndarray): The input image in which to detect markers.
        detector: The ArUco marker detector instance.
    
    Returns:
        numpy.ndarray: The image with ArUco markers whitened out.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the image
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        # Iterate over detected markers
        for marker_corners in corners:
            # Convert the corners to integer pixel positions
            pts = marker_corners[0].astype(np.int32)

            # Fill the marker region with white color (whitening the marker area)
            cv2.fillPoly(image, [pts], (255, 255, 255))
    
    return image

# Example usage
image_path = path_in  # Replace with your image path
image = cv2.imread(image_path)

# Load the ArUco dictionary and detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Step 1: Whiten the ArUco markers
image_without_markers, detected_ids, detected_corners = whiten_aruco_markers(image, detector)

# Step 2: (Optional) Display or save the result after whitening
cv2.imwrite(path_out, image_without_markers)

# Step 2: Show original image with detected markers for debugging
cv2.imshow("Original Image with Detected Markers", image)
if detected_ids is not None:
    print(f"Detected Marker IDs: {detected_ids.flatten()}")
    print(f"Detected Marker Corners: {detected_corners}")
else:
    print("No markers detected.")

# Step 3: Show the whitened image
cv2.imshow("Image without ArUco Markers", image_without_markers)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()