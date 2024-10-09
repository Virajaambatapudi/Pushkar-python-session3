import numpy as np

# Load the data from the .npz file
data_path = r'D:\\Python Tutorial\\Pushkar project-3\\road_signs_data.npz'
data = np.load(data_path)

# Extract images and labels
images = data['images']
labels = data['labels']

# Check the shapes of the loaded arrays
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
