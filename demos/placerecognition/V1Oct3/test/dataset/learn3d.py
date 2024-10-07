import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to extract frames from video
def extract_frames(video_path, max_frames=None):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            break
    cap.release()
    return frames

# Custom dataset for video frames
class VideoFrameDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        return self.transform(frame)

class BARF(nn.Module):
    def __init__(self, num_layers=8, hidden_dim=256, num_frames=100):
        super(BARF, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # NeRF MLP
        self.layers = nn.ModuleList([
            nn.Linear(3, hidden_dim)] + 
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers-2)] +
            [nn.Linear(hidden_dim, 4)]  # RGB + density
        )

        # Learnable camera poses
        self.poses = nn.Parameter(torch.randn(num_frames, 6))  # [R, t]

    def forward(self, x, frame_idx):
        # Apply pose transformation
        pose = self.poses[frame_idx]
        x = self.transform_points(x, pose)

        # NeRF forward pass
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = F.relu(x)

        rgb = torch.sigmoid(x[:, :3])
        density = F.relu(x[:, 3])

        return rgb, density

    def transform_points(self, points, pose):
        # Simple pose transformation (rotation + translation)
        rotation = pose[:3]
        translation = pose[3:]
        
        rotated = torch.einsum('ij,bj->bi', self.rotation_matrix(rotation), points)
        transformed = rotated + translation

        return transformed

    def rotation_matrix(self, rotation):
        # Convert axis-angle rotation to rotation matrix
        angle = torch.norm(rotation)
        axis = rotation / (angle + 1e-8)
        
        K = torch.tensor([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
        
        I = torch.eye(3)
        R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.matmul(K, K)
        
        return R

def train_barf(model, dataloader, num_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for i, frames in enumerate(dataloader):
            optimizer.zero_grad()

            # Generate random 3D points
            points = torch.rand(1000, 3) * 2 - 1  # Range: [-1, 1]

            # Forward pass
            rgb, density = model(points, i)

            # Compute loss (this is a placeholder - you'd need a more sophisticated loss)
            target = frames.view(-1, 3)  # Flatten the image
            loss = F.mse_loss(rgb, target)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

def visualize_camera_poses(model):
    # Extract camera positions and orientations
    positions = model.poses[:, 3:].detach().numpy()
    orientations = model.poses[:, :3].detach().numpy()

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot camera positions
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b', marker='o')

    # Plot camera orientations (as arrows)
    for pos, ori in zip(positions, orientations):
        ax.quiver(pos[0], pos[1], pos[2], ori[0], ori[1], ori[2], length=0.1, normalize=True, color='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Estimated Camera Poses')

    plt.show()

# Main execution
if __name__ == "__main__":
    # Set up data
    video_path = "/home/farmspace/ontouchai/demos/placerecognition/V1Oct3/test/dataset/workshop/general/VID_20241003_195637.mp4"  # Replace with your video path
    frames = extract_frames(video_path, max_frames=100)  # Limit to 100 frames for this example
    dataset = VideoFrameDataset(frames)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Initialize and train model
    model = BARF(num_frames=len(frames))
    train_barf(model, dataloader, num_epochs=50)  # Reduced epochs for demonstration

    # Visualize camera poses
    visualize_camera_poses(model)