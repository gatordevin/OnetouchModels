import os
import cv2
import random
import shutil
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Define input and output directories
parent_folder = "/home/farmspace/ontouchai/demos/terrainclassifier/V1Oct3/train/dataset/videos"
output_folder = "/home/farmspace/ontouchai/demos/terrainclassifier/V1Oct3/train/dataset/final"

# Define train and val output directories
train_output = os.path.join(output_folder, "train")
val_output = os.path.join(output_folder, "val")

# Create necessary directories
os.makedirs(train_output, exist_ok=True)
os.makedirs(val_output, exist_ok=True)

# Function to create class folders
def create_class_folders(base_folder, class_name):
    train_class_dir = os.path.join(base_folder, "train", class_name)
    val_class_dir = os.path.join(base_folder, "val", class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)
    return train_class_dir, val_class_dir

# Function to extract frames from a video
def extract_frames(video_path, output_dir, class_label, frame_skip=1, val_split=0.2):
    """
    Extract frames from a video and save them to the specified directory.

    Parameters:
    - video_path (str): Path to the video file.
    - output_dir (dict): Dictionary containing train and val output directories.
    - class_label (str): Label of the class for naming frames.
    - frame_skip (int): Extract every 'frame_skip' frames.
    - val_split (float): Proportion of frames to use for validation.
    
    Returns:
    - tuple: (train_count, val_count)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.warning(f"Cannot open video file {video_path}")
        return 0, 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    val_start_frame = int(frame_count * (1 - val_split))

    train_count = 0
    val_count = 0
    saved_count = 0

    for frame_num in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_skip == 0:
            frame_filename = f"{class_label}_{saved_count:06d}.jpg"
            if frame_num >= val_start_frame:
                output_path = os.path.join(output_dir['val'], frame_filename)
                val_count += 1
            else:
                output_path = os.path.join(output_dir['train'], frame_filename)
                train_count += 1
            try:
                cv2.imwrite(output_path, frame)
                saved_count += 1
            except Exception as e:
                logging.error(f"Failed to write frame {output_path}: {e}")

    cap.release()
    return train_count, val_count

# Initialize dictionaries to hold frame counts
train_frame_counts = defaultdict(int)
val_frame_counts = defaultdict(int)

# List all classes
classes = [d for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
logging.info(f"Found classes: {classes}")

# Extract frames from videos
for class_name in classes:
    class_dir = os.path.join(parent_folder, class_name)
    train_class_output, val_class_output = create_class_folders(output_folder, class_name)

    videos_dir = os.path.join(class_dir)
    if os.path.exists(videos_dir):
        videos = [f for f in os.listdir(videos_dir) if f.endswith(".mp4")]
        logging.info(f"Processing {len(videos)} videos for class '{class_name}'")
        for video in tqdm(videos, desc=f"Extracting Frames for {class_name}"):
            video_path = os.path.join(videos_dir, video)
            output_dirs = {'train': train_class_output, 'val': val_class_output}
            train_extracted, val_extracted = extract_frames(video_path, output_dirs, class_name, frame_skip=1, val_split=0.2)
            train_frame_counts[class_name] += train_extracted
            val_frame_counts[class_name] += val_extracted
    else:
        logging.warning(f"Videos directory does not exist: {videos_dir}")

logging.info("\nFrame extraction complete.")

# Function to balance classes
def balance_classes(base_folder, frame_counts):
    """
    Balance the number of frames per class by limiting to the minimum frame count.

    Parameters:
    - base_folder (str): Base directory ('train' or 'val').
    - frame_counts (dict): Dictionary with class names as keys and frame counts as values.
    """
    if not frame_counts:
        logging.warning(f"No frames found in '{base_folder}' to balance.")
        return

    min_frames = min(frame_counts.values())
    logging.info(f"Minimum frames per class in '{base_folder}': {min_frames}")

    for class_name, count in frame_counts.items():
        class_dir = os.path.join(base_folder, class_name)
        all_frames = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
        if count > min_frames:
            selected_frames = set(random.sample(all_frames, min_frames))
            # Remove extra frames
            extra_frames = set(all_frames) - selected_frames
            for frame in extra_frames:
                try:
                    os.remove(os.path.join(class_dir, frame))
                except Exception as e:
                    logging.error(f"Failed to remove frame {frame}: {e}")
            logging.info(f"Class '{class_name}': Reduced from {count} to {min_frames} frames.")
        else:
            logging.info(f"Class '{class_name}': {count} frames (no reduction needed).")

# Balance train and val classes
balance_classes(train_output, train_frame_counts)
balance_classes(val_output, val_frame_counts)

logging.info("\nClasses balanced for both training and validation sets.")

# Save class names to a JSON file for future use
final_results_dir = "/home/farmspace/ontouchai/demos/terrainclassifier/V1Oct3/train/results"
os.makedirs(final_results_dir, exist_ok=True)
class_names_path = os.path.join(final_results_dir, 'class_names.json')
try:
    with open(class_names_path, 'w') as f:
        json.dump(classes, f)
    logging.info(f"Class names saved to '{class_names_path}'")
except Exception as e:
    logging.error(f"Failed to save class names: {e}")

# Print dataset statistics
def print_dataset_stats(folder):
    total_images = 0
    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            num_images = len([f for f in os.listdir(class_path) if f.endswith('.jpg')])
            logging.info(f"{class_name}: {num_images} images")
            total_images += num_images
    logging.info(f"Total images in {os.path.basename(folder)}: {total_images}")

logging.info("\nTraining set statistics:")
print_dataset_stats(train_output)
logging.info("\nValidation set statistics:")
print_dataset_stats(val_output)

# Additional Suggestions:
# - Frame Skipping: To reduce dataset size and prevent redundancy, consider increasing 'frame_skip'.
# - Ensure No Data Leakage: Verify that frames from the same video are not split between train and val sets.
# - Random Seed: A random seed is set for reproducibility.