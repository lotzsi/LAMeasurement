import os
from PIL import Image, ImageEnhance, ImageStat

# Function to calculate the brightness of an image
def calculate_brightness(image):
    # Convert image to grayscale
    grayscale_image = image.convert('L')
    # Calculate the mean brightness
    stat = ImageStat.Stat(grayscale_image)
    return stat.mean[0]

# Function to adjust brightness based on a given factor
def adjust_brightness(image_path, output_path, adjustment_factor):
    with Image.open(image_path) as img:
        # Initialize brightness enhancer
        enhancer = ImageEnhance.Brightness(img)
        # Adjust brightness
        img_enhanced = enhancer.enhance(adjustment_factor)
        # Save the enhanced image
        img_enhanced.save(output_path)

# Set the folder paths
input_folder = 'Bilder_Blätter_LAI/Beispiel'
output_folder = 'Bilder_Blätter_LAI/Beispiel/AdjustedBrightness'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all the image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]

# Open the first image as the reference image
reference_image_path = 'Bilder_Blätter_LAI/LitterTraps_240930/REF.JPG'
with Image.open(reference_image_path) as reference_image:
    reference_brightness = calculate_brightness(reference_image)
    print(f"Reference brightness: {reference_brightness}")

# Loop through all images in the folder and adjust their brightness
for filename in image_files:
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    # Open the image and calculate its brightness
    with Image.open(input_path) as img:
        image_brightness = calculate_brightness(img)
        print(f"Processing {filename}: current brightness {image_brightness}")

        # Calculate the adjustment factor to match the reference brightness
        if image_brightness != 0:  # Avoid division by zero
            adjustment_factor = reference_brightness / image_brightness
        else:
            adjustment_factor = 1.0  # If brightness is 0 (completely dark), we don't adjust

        print(f"Adjustment factor for {filename}: {adjustment_factor}")

        # Adjust brightness and save the new image
        adjust_brightness(input_path, output_path, adjustment_factor)

print("All images have been adjusted to the same brightness!")