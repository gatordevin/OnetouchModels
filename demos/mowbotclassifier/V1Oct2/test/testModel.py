import os
import torch
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import json
import torch.nn as nn
import torch.nn.functional as F  # Import F to avoid 'F' not defined errors

# ============================
# 1. Device Configuration
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# ============================
# 2. Define ResNetWithEmbedding Model
# ============================
class ResNetWithEmbedding(nn.Module):
    def __init__(self, num_classes):
        super(ResNetWithEmbedding, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        
        # Freeze all layers initially
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze layer4 for fine-tuning
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        
        # Replace the last fully connected layer with an embedding layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 128)
        
        # Classification head
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Extract features
        features = self.resnet(x)  # [batch_size, 128]
        logits = self.classifier(features)  # [batch_size, num_classes]
        return logits, features

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
model = ResNetWithEmbedding(num_classes=num_classes)
model = model.to(device)

model_path = os.path.join(results_dir, 'best_model_resnet18.pth')  # Ensure this path matches the training script

if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    exit()

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    print(f"Loaded model weights from '{model_path}'")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# ============================
# 5. Define Transforms
# ============================
# Use the same normalization as during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Match training resize
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])  # ImageNet standards used during training
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
        logits, _ = model(image_tensor)
        _, preds = torch.max(logits, 1)
    return preds.item()

# ============================
# 7. Video Processing
# ============================
input_video_path = '/home/farmspace/ontouchai/demos/mowbotclassifier/V1Oct2/test/videos/VID_20241003_143825.mp4'  # Replace with your input video path
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
    # Adjust class names as per your dataset. Ensure consistency in casing.
    if pred_class.lower() == 'weeds':
        color = (0, 0, 255)  # Red for 'weeds' to indicate caution
    elif pred_class.lower() == 'tall_grass':
        color = (255, 0, 0)  # Blue for 'tall_grass'
    elif pred_class.lower() == 'short_grass':
        color = (0, 255, 0)  # Green for 'short_grass'
    else:
        color = (255, 255, 255)  # White for other classes
    
    # Add a filled rectangle behind the text for better visibility
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (10, 10), (10 + text_width + 10, 10 + text_height + 10), color, -1)
    
    # Put text over the rectangle
    cv2.putText(frame, label, (15, 15 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Write the frame to the output video
    out.write(frame)

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video saved to '{output_video_path}'")
