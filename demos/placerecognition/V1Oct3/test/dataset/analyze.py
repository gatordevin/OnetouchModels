import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import os

BASE_PATH = '/home/farmspace/ontouchai/demos/placerecognition/V1Oct3/test/dataset'
RESULTS_PATH = os.path.join(BASE_PATH, 'results')
VISUALIZATIONS_PATH = os.path.join(RESULTS_PATH, 'visualizations')
# Load the saved features
# Load the saved features
features = np.load(os.path.join(RESULTS_PATH, 'spatial_feature_embeddings.npy'))

# 1. Preprocess: Calculate frame-to-frame differences
frame_differences = np.diff(features, axis=0)

# 2. Use PCA to extract 6 principal components
pca = PCA(n_components=6)
motion_components = pca.fit_transform(frame_differences)

# Define labels for our 6 degrees of freedom
dof_labels = ['X Translation', 'Y Translation', 'Z Translation', 
              'X Rotation', 'Y Rotation', 'Z Rotation']

# 3. Analyze and visualize components

# 3.1 Plot all 6 motion components
plt.figure(figsize=(15, 10))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.plot(motion_components[:, i])
    plt.title(dof_labels[i])
    plt.xlabel('Frame')
    plt.ylabel('Component Magnitude')
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATIONS_PATH, 'six_dof_components.png'))
plt.close()

# 3.2 Compute cumulative motion for each component
cumulative_motion = np.cumsum(motion_components, axis=0)

# Smooth the paths
window_length = 51  # Must be odd; adjust based on your video length
poly_order = 3
smoothed_path = savgol_filter(cumulative_motion, window_length, poly_order, axis=0)

# 3.3 Plot smoothed cumulative motion for translation and rotation separately
fig = plt.figure(figsize=(20, 8))

# Translation
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(smoothed_path[:, 0], smoothed_path[:, 1], smoothed_path[:, 2])
scatter = ax1.scatter(smoothed_path[:, 0], smoothed_path[:, 1], smoothed_path[:, 2],
                      c=range(len(smoothed_path)), cmap='viridis')
ax1.set_title('Cumulative Translation Path')
ax1.set_xlabel('X Translation')
ax1.set_ylabel('Y Translation')
ax1.set_zlabel('Z Translation')
fig.colorbar(scatter, ax=ax1, label='Frame Number')

# Rotation
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(smoothed_path[:, 3], smoothed_path[:, 4], smoothed_path[:, 5])
scatter = ax2.scatter(smoothed_path[:, 3], smoothed_path[:, 4], smoothed_path[:, 5],
                      c=range(len(smoothed_path)), cmap='viridis')
ax2.set_title('Cumulative Rotation Path')
ax2.set_xlabel('X Rotation')
ax2.set_ylabel('Y Rotation')
ax2.set_zlabel('Z Rotation')
fig.colorbar(scatter, ax=ax2, label='Frame Number')

plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATIONS_PATH, 'cumulative_6dof_motion.png'))
plt.close()

# 3.4 Analyze motion magnitudes
motion_magnitudes = np.abs(motion_components)

plt.figure(figsize=(15, 8))
for i in range(6):
    plt.plot(motion_magnitudes[:, i], label=dof_labels[i])
plt.title('Motion Magnitudes Over Time')
plt.xlabel('Frame')
plt.ylabel('Magnitude')
plt.legend()
plt.savefig(os.path.join(VISUALIZATIONS_PATH, 'six_dof_magnitudes.png'))
plt.close()

# 4. Analyze dominant motion types
dominant_motion = np.argmax(motion_magnitudes, axis=1)

plt.figure(figsize=(15, 8))
for i in range(6):
    frames = np.where(dominant_motion == i)[0]
    plt.scatter(frames, [i] * len(frames), alpha=0.5, label=dof_labels[i])
plt.title('Dominant Motion Type Over Time')
plt.xlabel('Frame')
plt.ylabel('Motion Type')
plt.yticks(range(6), dof_labels)
plt.legend()
plt.savefig(os.path.join(VISUALIZATIONS_PATH, 'dominant_motion_type.png'))
plt.close()

# 5. Compute and visualize motion energy distribution
motion_energy = np.sum(motion_magnitudes**2, axis=0)
motion_energy_percentage = motion_energy / np.sum(motion_energy) * 100

plt.figure(figsize=(10, 6))
plt.bar(dof_labels, motion_energy_percentage)
plt.title('Distribution of Motion Energy')
plt.xlabel('Degree of Freedom')
plt.ylabel('Percentage of Total Motion Energy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATIONS_PATH, 'motion_energy_distribution.png'))
plt.close()

print("Six degrees of freedom motion analysis completed. Visualizations saved in:", VISUALIZATIONS_PATH)