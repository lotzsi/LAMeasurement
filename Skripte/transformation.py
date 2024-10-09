# dependencies
import cv2
import numpy as np


# input and output file names
file_name_in = 'DSC03398_wb-2.JPG'
file_name_out = 'DSC03398_warped-2.JPG'

path_in = '/Users/simon/Documents/Master/Masterarbeit/aktuelles Thema/Bilder_Blätter_LAI/Beispiel/'
path_out = path_in

# horizontal and vertical distances between ArUco markers (mm, outer corners)
# algorithm sorts corner ids (ascending) and transforms images that lowest id is at upper-left corner. No manual image rotations necessary 
#width_mm = 197
width_mm = 900
#height_mm = 286
height_mm = 1900

# read image
image = cv2.imread(path_in + file_name_in)

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
        #max_index = np.argmax(projections)
        min_index = np.argmin(projections)
        large_quad_corners.append(points[min_index])

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

    # write image
    cv2.imwrite(path_out + file_name_out, warped)
else:
    print("Not all ArUco markers detected.")
    print(len(corners))

print("done")