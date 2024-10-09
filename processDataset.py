import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

images_dir = r'D:\\Python Tutorial\\Pushkar project-3\\archive\\images' #path of the images
annotations_dir = r'D:\\Python Tutorial\\Pushkar project-3\\archive\\annotations' #path of the annotations

image_size = (64, 64)

images = []
labels = []


# Function to parse the XML file for labels
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    label = root.find('object').find('name').text
    return label


# Process images and labels
for filename in os.listdir(images_dir):
    if filename.endswith('.png'): 
        # Load the image
        img_path = os.path.join(images_dir, filename)
        img = cv2.imread(img_path)

        # Resize the image
        img_resized = cv2.resize(img, image_size)

        # Convert image to array
        img_array = np.array(img_resized)
        images.append(img_array)

        # Get corresponding XML file
        xml_file = os.path.join(annotations_dir, filename.replace('.png', '.xml'))
        if os.path.exists(xml_file):
            label = parse_xml(xml_file)
            labels.append(label)

# Convert lists to Numpy arrays
images = np.array(images)
labels = np.array(labels)

# Save the processed images and labels as a .npz file
np.savez('road_signs_data.npz', images=images, labels=labels)

# Check the shapes of the resulting arrays
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
