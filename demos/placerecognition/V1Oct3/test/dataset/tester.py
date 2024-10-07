import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import os
from learn3d import DisentangledVAE
from posfromembed import DisentangledAutoencoder
from matplotlib import pyplot as plt
# Load the VAE and Autoencoder models
@st.cache_resource
def load_models(vae_path, autoencoder_path):
    vae_model = DisentangledVAE(image_size=128, latent_size=128)
    vae_model.load_state_dict(torch.load(vae_path, map_location=torch.device('cpu')))
    vae_model.eval()

    autoencoder_model = DisentangledAutoencoder(input_dim=128, latent_dim=6)
    autoencoder_model.load_state_dict(torch.load(autoencoder_path, map_location=torch.device('cpu')))
    autoencoder_model.eval()

    return vae_model, autoencoder_model

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Encode image using VAE
def encode_image(model, image_tensor):
    with torch.no_grad():
        _, mu, _ = model(image_tensor)
    return mu.squeeze().numpy()

# Encode and decode using the autoencoder
def process_autoencoder(model, latent_vector):
    with torch.no_grad():
        latent_tensor = torch.FloatTensor(latent_vector).unsqueeze(0)
        _, disentangled = model(latent_tensor)
    return disentangled.squeeze().numpy()

# Decode latent vector using VAE
def decode_latent(model, latent_vector):
    with torch.no_grad():
        latent_tensor = torch.FloatTensor(latent_vector).unsqueeze(0)
        decoded = model.decoder(latent_tensor)
    return decoded.squeeze().permute(1, 2, 0).numpy()

# Load video frames
def load_video_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

# Main Streamlit app
def main():
    st.title("Dual-Model VAE Latent Space Explorer")

    # Load the models
    vae_path = st.text_input("Enter the path to your VAE model:", "/home/farmspace/ontouchai/demos/placerecognition/V1Oct3/test/dataset/results/disentangled_vae_model.pth")
    autoencoder_path = st.text_input("Enter the path to your Autoencoder model:", "/home/farmspace/ontouchai/demos/placerecognition/V1Oct3/test/dataset/results/disentangled_autoencoder_model.pth")
    if not vae_path or not autoencoder_path:
        st.warning("Please enter valid paths for both models.")
        return
    
    vae_model, autoencoder_model = load_models(vae_path, autoencoder_path)

    # Video selection
    video_folder = st.text_input("Enter the path to your video folder:", "/home/farmspace/ontouchai/demos/placerecognition/V1Oct3/test/dataset/workshop/forward")
    if not os.path.exists(video_folder):
        st.warning("Please enter a valid video folder path.")
        return

    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    selected_video = st.selectbox("Choose a video:", video_files)
    video_path = os.path.join(video_folder, selected_video)

    # Load and display video frames
    frames = load_video_frames(video_path)
    selected_frame_index = st.slider("Select a frame:", 0, len(frames) - 1)
    selected_frame = frames[selected_frame_index]
    st.image(selected_frame, caption="Selected Frame", use_column_width=True)

    # Process the selected frame
    image = Image.fromarray(selected_frame)
    image_tensor = preprocess_image(image)
    vae_latent = encode_image(vae_model, image_tensor)
    disentangled_latent = process_autoencoder(autoencoder_model, vae_latent)

    # Create sliders for latent variables
    st.subheader("Adjust Disentangled Latent Variables")
    adjusted_latent = disentangled_latent.copy()
    cols = st.columns(3)
    for i, name in enumerate(['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']):
        with cols[i % 3]:
            adjusted_latent[i] = st.slider(f"{name}", -3.0, 3.0, float(disentangled_latent[i]), step=0.1)

    # Reconstruct and display the adjusted image
    if st.button("Update Image"):
        # Encode back to VAE latent space
        reconstructed_vae_latent = autoencoder_model.decoder(torch.FloatTensor(adjusted_latent).unsqueeze(0)).squeeze().detach().numpy()
        
        # Decode using VAE
        decoded_image = decode_latent(vae_model, reconstructed_vae_latent)
        st.image(decoded_image, caption="Reconstructed Image", use_column_width=True)

    # Display latent space visualization
    st.subheader("Disentangled Latent Space Visualization")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(adjusted_latent)), adjusted_latent)
    ax.set_xlabel("Latent Dimensions")
    ax.set_ylabel("Value")
    ax.set_title("Disentangled Latent Space Representation")
    st.pyplot(fig)

if __name__ == "__main__":
    main()