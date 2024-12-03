import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.draw import polygon, disk
from tqdm import tqdm

seed = 1234
np.random.seed(seed)
random.seed(seed)

# Initialize dimensions
H, W = 18, 18

# Triangle (hollow with thicker lines)
triangle = np.zeros((H, W), dtype=np.float32)

outer_rr, outer_cc = polygon([4, 11, 11], [8, 1, 15], shape=(H, W))
inner_rr, inner_cc = polygon([6, 9, 9], [8, 5, 11], shape=(H, W))
triangle[outer_rr, outer_cc] = 1.0
triangle[inner_rr, inner_cc] = 0.0

# Square (hollow with thicker lines)
square = np.zeros((H, W), dtype=np.float32)
square[3:13, 3:13] = 1.0  # Outer square
square[5:11, 5:11] = 0.0  # Inner square

# Circle (hollow, larger)
circle = np.zeros((H, W), dtype=np.float32)
outer_rr, outer_cc = disk((8, 8), 7, shape=(H, W))
inner_rr, inner_cc = disk((8, 8), 5, shape=(H, W))
circle[outer_rr, outer_cc] = 1.0
circle[inner_rr, inner_cc] = 0.0

# Diamond (hollow, larger and thicker lines)
diamond = np.zeros((H, W), dtype=np.float32)
outer_rr, outer_cc = polygon([8, 2, 8, 14], [2, 8, 14, 8], shape=(H, W))
inner_rr, inner_cc = polygon([8, 4, 8, 12], [4, 8, 12, 8], shape=(H, W))
diamond[outer_rr, outer_cc] = 1.0
diamond[inner_rr, inner_cc] = 0.0

# Shape dictionary
shapes = {1: triangle, 2: square, 3: circle, 4: diamond}

# Parameters
image_size = 40
num_images = 42000
min_shapes = 2
max_shapes = 4
padding = 0

# Storage for images and labels
images = np.zeros((num_images, image_size, image_size), dtype=np.float32)
class_labels = []  # Class label for each pixel
instance_labels = []  # Instance label for each pixel

# Function to place a shape in an image with bounds checking
def place_shape(image, class_label, instance_label, shape, shape_id, instance_id, pos_x, pos_y):
    for i in range(H):
        for j in range(W):
            if shape[i, j] == 1.0:
                xi, yj = pos_x + i, pos_y + j
                if class_label[xi, yj] == 0:  # If background, place shape
                    image[xi, yj] = 1.0  # Shape pixel value
                    class_label[xi, yj] = shape_id
                    instance_label[xi, yj] = instance_id
                else:  # Overlapping area
                    class_label[xi, yj] = -1
                    instance_label[xi, yj] = -1
                    image[xi, yj] = 1.0  # Ensure overlapping area is also bright

# Generate images and labels
for img_index in tqdm(range(num_images)):
    # Generate a random background color between 0 and 0.3
    bg_color = random.uniform(0.1, 0.6)
    
    # Initialize the image with the random background color
    image = np.full((image_size, image_size), bg_color, dtype=np.float32)
    class_label = np.zeros((image_size, image_size), dtype=np.int8)
    instance_label = np.zeros((image_size, image_size), dtype=np.int8)
    
    # Randomly select number of shapes
    num_shapes = random.randint(min_shapes, max_shapes)
    
    # Randomly place shapes
    for instance_id in range(1, num_shapes + 1):
        shape_id = random.randint(1, 4)  # Randomly select a shape
        shape = shapes[shape_id]
    
        # Random position with padding
        pos_x = random.randint(padding, image_size - H - padding)
        pos_y = random.randint(padding, image_size - W - padding)
    
        # Place shape
        place_shape(image, class_label, instance_label, shape, shape_id, instance_id, pos_x, pos_y)
    
    # Store the image and labels
    images[img_index] = image
    class_labels.append(class_label)
    instance_labels.append(instance_label)

# Convert class_labels and instance_labels lists to numpy arrays for easy saving
class_labels = np.array(class_labels, dtype=np.int8)
instance_labels = np.array(instance_labels, dtype=np.int8)

# Example of displaying one generated image with labels
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
ax1.imshow(images[0], cmap="gray", vmin=0, vmax=1)
ax1.set_title(f"Generated Image (Background color: {images[0][0,0]:.2f})")
ax2.imshow(class_labels[0], cmap="gray")
ax2.set_title("Class Label (0: background, -1: overlap, 1-4: shapes)")
ax3.imshow(instance_labels[0], cmap="nipy_spectral")
ax3.set_title("Instance Label (-1: overlap)")
plt.show()

train = np.arange(0, 40000)
val = np.arange(40000, 41000)
test = np.arange(41000, 42000)

images = images[:, None]
# Saving images and labels (optional)
import os
os.makedirs("./Shapes", exist_ok=True)
dataset = {"images":images[train], "labels": instance_labels[train], "pixelwise_class_labels": class_labels[train]}
np.savez_compressed("./Shapes/train.npz", **dataset)
dataset = {"images":images[val], "labels": instance_labels[val], "pixelwise_class_labels": class_labels[val]}
np.savez_compressed("./Shapes/val.npz", **dataset)
dataset = {"images":images[test], "labels": instance_labels[test], "pixelwise_class_labels": class_labels[test]}
np.savez_compressed("./Shapes/test.npz", **dataset)
                                                                                                                             
# np.save("instance_labels.npy", instance_labels)
