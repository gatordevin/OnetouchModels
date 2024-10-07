import os
import torch
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import json

# ============================
# 1. Device Configuration
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# ============================
# 2. Define ResNet18 Model with Modified Classifier
# ============================
def get_resnet18_model(num_classes):
    """
    Initializes a ResNet18 model with a modified classifier.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: Modified ResNet18 model.
    """
    model = models.resnet18(pretrained=False)  # Load without pre-trained weights
    # Modify the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_ftrs, 128),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(128, num_classes)
    )
    return model

# ============================
# 3. Load Class Names
# ============================
results_dir = '/home/farmspace/ontouchai/demos/mowbotclassifier/V1Oct2/train/results'
class_names_path = os.path.join(results_dir, 'class_names.json')

if not os.path.exists(class_names_path):
    print(f"Error: Class names file '{class_names_path}' not found.")
    exit()

with open(class_names_path, 'r') as f:
    class_names = json.load(f)

num_classes = len(class_names)
print(f'Loaded class names: {class_names}')

# ============================
# 4. Initialize and Load the Model
# ============================
model = get_resnet18_model(num_classes).to(device)
model_path = os.path.join(results_dir, 'best_model_resnet18.pth')  # Updated model path

if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    exit()

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model weights from '{model_path}'")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# ============================
# 5. Define Transforms
# ============================
# Use the same normalization as during training
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================
# 6. Define Prediction Function
# ============================
def predict(image_tensor):
    """
    Predicts the class of an input image tensor.

    Args:
        image_tensor (torch.Tensor): Preprocessed image tensor.

    Returns:
        int: Predicted class index.
    """
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
    return preds.item()

# ============================
# 7. Video Processing
# ============================
input_video_path = '/home/farmspace/ontouchai/demos/mowbotclassifier/V1Oct2/test/videos/VID_20241002_180435.mp4'  # Replace with your input video path
output_video_path = '/home/farmspace/ontouchai/demos/mowbotclassifier/V1Oct2/test/results/video_with_labels.mp4'  # Replace with desired output path

# Check if input video exists
if not os.path.exists(input_video_path):
    print(f"Error: Input video file '{input_video_path}' not found.")
    exit()

# Open the input video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video file '{input_video_path}'")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as needed
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print(f"Processing video... Total frames: {total_frames}")

for _ in tqdm(range(total_frames), desc="Processing Frames"):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    pil_image = Image.fromarray(image)  # Convert to PIL Image
    image_tensor = transform(pil_image)  # Apply transforms
    
    # Predict
    pred_idx = predict(image_tensor)
    pred_class = class_names[pred_idx]
    
    # Overlay label on frame
    label = f"{pred_class}"
    # Choose color based on prediction
    if pred_class == 'weeds':
        color = (0, 0, 255)  # Red for 'weeds' to indicate caution
    elif pred_class == 'tall_grass':
        color = (255, 0, 0)  # Blue for 'tall_grass'
    elif pred_class == 'short_grass':
        color = (0, 255, 0)  # Green for 'short_grass'
    else:
        color = (255, 255, 255)  # White for other classes
    
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, color, 2, cv2.LINE_AA)
    
    # Optionally, add a rectangle around the label for better visibility
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (5, 5), (15 + text_width, 35 + text_height), color, -1)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Write the frame to the output video
    out.write(frame)

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video saved to '{output_video_path}'")
