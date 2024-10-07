import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from tqdm import tqdm

import sys
from learn3d import DisentangledVAE

# Assume the DisentangledVAE is already defined and loaded

# Path setup
BASE_PATH = '/home/farmspace/ontouchai/demos/placerecognition/V1Oct3/test/dataset'
WORKSHOP_PATH = os.path.join(BASE_PATH, 'workshop')
MODEL_PATH = os.path.join(BASE_PATH, 'results', 'disentangled_vae_model.pth')
NEW_MODEL_PATH = os.path.join(BASE_PATH, 'results', 'disentangled_autoencoder_model.pth')

# Load the pre-trained DisentangledVAE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae_model = DisentangledVAE(image_size=128, latent_size=128)
vae_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
vae_model.to(device)
vae_model.eval()

# New Autoencoder for disentanglement
class DisentangledAutoencoder(nn.Module):
    def __init__(self, input_dim=128, latent_dim=6):
        super(DisentangledAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# Dataset for the new autoencoder
class EmbeddingDataset(Dataset):
    def __init__(self, folder_path, vae_model, device):
        self.embeddings = []
        self.labels = []
        self.categories = ['down', 'forward', 'left', 'pitch', 'roll', 'yaw']
        
        for category in self.categories:
            category_path = os.path.join(folder_path, category)
            video_path = os.path.join(category_path, os.listdir(category_path)[0])  # Assume one video per category
            
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (128, 128))
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                frames.append(frame)
            cap.release()
            
            frames = torch.stack(frames).to(device)
            with torch.no_grad():
                _, mu, _ = vae_model(frames)
            
            self.embeddings.extend(mu.cpu().numpy())
            self.labels.extend([self.categories.index(category)] * len(mu))
        
        self.embeddings = np.array(self.embeddings)
        self.labels = np.array(self.labels)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.embeddings[idx]), torch.LongTensor([self.labels[idx]])

# Loss function
def disentanglement_loss(reconstructed, original, latent, label):
    reconstruction_loss = nn.MSELoss()(reconstructed, original)
    
    batch_size, latent_dim = latent.size()
    
    # Create a one-hot encoding of the labels
    label_one_hot = torch.zeros(batch_size, latent_dim, device=latent.device)
    label_one_hot.scatter_(1, label, 1)
    
    # Encourage the corresponding latent variable to be high for the correct category
    category_loss = torch.mean(torch.sum((1 - latent * label_one_hot)**2, dim=1))
    
    # Encourage other latent variables to be low
    other_loss = torch.mean(torch.sum((latent * (1 - label_one_hot))**2, dim=1))
    
    total_loss = reconstruction_loss + 0.1 * category_loss + 0.1 * other_loss
    
    return total_loss, reconstruction_loss, category_loss, other_loss

# Training function
def train_disentangled_autoencoder(model, dataloader, num_epochs, device):
    # In your training loop:
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reduced learning rate

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (embeddings, labels) in enumerate(dataloader):
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            optimizer.zero_grad()
            reconstructed, latent = model(embeddings)
            loss, recon_loss, cat_loss, other_loss = disentanglement_loss(reconstructed, embeddings, latent, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, "
                    f"Cat: {cat_loss.item():.4f}, Other: {other_loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Main execution
if __name__ == "__main__":
    batch_size = 32
    num_epochs = 300
    
    dataset = EmbeddingDataset(WORKSHOP_PATH, vae_model, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = DisentangledAutoencoder().to(device)
    
    train_disentangled_autoencoder(model, dataloader, num_epochs, device)

    # Visualization
    model.eval()
    with torch.no_grad():
        sample_embedding, sample_label = dataset[0]
        sample_embedding = sample_embedding.unsqueeze(0).to(device)
        reconstructed, latent = model(sample_embedding)
        
        print("Original embedding:", sample_embedding.cpu().numpy())
        print("Reconstructed embedding:", reconstructed.cpu().numpy())
        print("Disentangled latent space:", latent.cpu().numpy())
        print("Corresponding category:", dataset.categories[sample_label.item()])