import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Directory paths
image_dir = 'images'  # Replace with your images directory path
annotation_dir = 'annotations'  # Replace with your annotations directory path

# Image settings
image_size = (224, 224)  # Resize images to 224x224 pixels

# Lists to store image data and labels
images = []
labels = []

# Function to parse XML and extract bounding box data
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    for member in root.findall('object'):
        bndbox = member.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])

    return np.array(boxes)  # Return all bounding boxes

# Function to load and preprocess images
def load_images_and_labels(image_dir, annotation_dir):
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            # Load image
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)
            img = img / 255.0  # Normalize pixel values
            images.append(img)

            # Load corresponding annotation
            xml_path = os.path.join(annotation_dir, filename.replace('.jpg', '.xml'))
            label = parse_xml(xml_path)
            labels.append(label)

    # Convert lists to Numpy arrays
    images_np = np.array(images)
    labels_np = np.array(labels)
    return images_np, labels_np

# Load the images and labels
images_np, labels_np = load_images_and_labels(image_dir, annotation_dir)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images_np, labels_np, test_size=0.2, random_state=42)

# Define a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4)  # Assuming the output is bounding box coordinates (xmin, ymin, xmax, ymax)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
