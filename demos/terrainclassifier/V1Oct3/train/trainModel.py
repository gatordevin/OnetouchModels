import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import random
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
import warnings
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json  # For saving class names
import sys  # For tqdm
import logging  # For detailed logging

warnings.filterwarnings("ignore")

# ============================
# 1. Logging Configuration
# ============================
results_dir = '/home/farmspace/ontouchai/demos/terrainclassifier/V1Oct3/train/results'
os.makedirs(results_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(results_dir, 'training_log.txt'),
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

# ============================
# 2. Set Random Seeds for Reproducibility
# ============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ============================
# 3. Device Configuration
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f'Using device: {device}')
print(f'Using device: {device}')

# ============================
# 4. Data Loading and Augmentation
# ============================
data_dir = '/home/farmspace/ontouchai/demos/terrainclassifier/V1Oct3/train/dataset/final'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# Define transforms with 224x224 image size and optimized data augmentation
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Reduced image size
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    # Reduced Gaussian Blur parameters
    transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 3)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet standards
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Reduced image size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet standards
])

# Load datasets
logging.info("Loading datasets...")
print("Loading datasets...")
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)

class_names = train_dataset.classes
num_classes = len(class_names)
logging.info(f'Number of classes: {num_classes}')
logging.info(f'Classes: {class_names}')
print(f'Number of classes: {num_classes}')
print(f'Classes: {class_names}')

# Save class names to a JSON file for future use
class_names_path = os.path.join(results_dir, 'class_names.json')
with open(class_names_path, 'w') as f:
    json.dump(class_names, f)
logging.info(f"Class names saved to '{class_names_path}'")
print(f"Class names saved to '{class_names_path}'")

# Create data loaders
batch_size = 32  # Adjust based on GPU memory
num_workers = 4  # Increased from 0 to utilize multiple CPU cores
pin_memory = True  # Enabled for faster data transfer to GPU

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)

logging.info(f"Number of training samples: {len(train_dataset)}")
logging.info(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")

# Test loading a single batch
try:
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    logging.info(f"Successfully loaded a batch of {images.size(0)} training images.")
    print(f"Successfully loaded a batch of {images.size(0)} training images.")
except Exception as e:
    logging.error(f"Error loading training data: {e}")
    print(f"Error loading training data: {e}")
    sys.exit(1)

# ============================
# 5. Define Custom Loss Function
# ============================
class CustomLoss(nn.Module):
    def __init__(self, base_loss, penalty_weights, target_class_idxs):
        """
        Custom loss function that combines the base loss with additional penalties
        for falsely classifying non-target classes as target classes.

        Args:
            base_loss (nn.Module): The base loss function (e.g., CrossEntropyLoss).
            penalty_weights (dict): A dictionary mapping target class indices to their penalty weights.
            target_class_idxs (list): List of class indices to apply penalties on.
        """
        super(CustomLoss, self).__init__()
        self.base_loss = base_loss
        self.penalty_weights = penalty_weights  # {class_idx: weight}
        self.target_class_idxs = target_class_idxs  # [class_idx1, class_idx2, ...]

    def forward(self, outputs, targets):
        # Compute base loss
        ce_loss = self.base_loss(outputs, targets)
        
        # Compute softmax probabilities
        probs = torch.softmax(outputs, dim=1)
        
        # Initialize penalty
        penalty = 0.0
        
        for class_idx, weight in self.penalty_weights.items():
            # Probability assigned to the target class
            class_prob = probs[:, class_idx]
            
            # Mask for samples that are NOT the target class
            non_class_mask = (targets != class_idx).float()
            
            # Penalty: Encourage lower probability for target class on non-target samples
            penalty += weight * torch.mean(class_prob * non_class_mask)
        
        # Total loss is base loss plus penalty
        total_loss = ce_loss + penalty
        return total_loss

# ============================
# 6. Initialize the Model
# ============================
logging.info("Initializing the model...")
print("Initializing the model...")
model = models.resnet18(pretrained=True)

# Freeze all layers initially
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4 for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True

# Modify the final fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(128, num_classes)
)

model = model.to(device)
logging.info("Model initialized and moved to device.")
print("Model initialized and moved to device.")

# ============================
# 7. Define Loss Function and Optimizer
# ============================
# Assign higher weights to 'weeds' (class index 3) and 'tall_grass' (class index 2)
class_weights = torch.ones(num_classes).to(device)
# class_weights[3] = 2.0  # Higher weight for 'weeds'
# class_weights[2] = 1.5  # Higher weight for 'tall_grass'

# Base loss is CrossEntropyLoss with class weights
base_criterion = nn.CrossEntropyLoss(weight=class_weights)

# Define penalty weights for false positives
penalty_weights = {
    # 3: 5.0,  # 'weeds'
    # 2: 3.0   # 'tall_grass'
}

# Initialize CustomLoss
criterion = CustomLoss(base_loss=base_criterion, penalty_weights=penalty_weights, target_class_idxs=[3, 2])

# Only parameters of layer4 and fc are being optimized
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001, weight_decay=1e-4)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# ============================
# 8. Training Parameters
# ============================
num_epochs = 100
patience = 15
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
epochs_no_improve = 0

train_losses, val_losses = [], []
val_accuracies = []

# ============================
# 9. Training and Validation Functions
# ============================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    logging.info("Starting training epoch...")
    print("Starting training epoch...")
    for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc="Training", file=sys.stdout)):
        if batch_idx % 100 == 0:
            logging.info(f"Training batch {batch_idx}/{len(loader)}")
            print(f"Training batch {batch_idx}/{len(loader)}")
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    logging.info(f"Training epoch loss: {epoch_loss:.4f}")
    print(f"Training epoch loss: {epoch_loss:.4f}")
    return epoch_loss

def validate_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    all_preds = []
    all_labels = []

    logging.info("Starting validation epoch...")
    print("Starting validation epoch...")
    for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc="Validating", file=sys.stdout)):
        if batch_idx % 100 == 0:
            logging.info(f"Validation batch {batch_idx}/{len(loader)}")
            print(f"Validation batch {batch_idx}/{len(loader)}")
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double() / len(loader.dataset)

    logging.info(f"Validation epoch loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    print(f"Validation epoch loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc, all_preds, all_labels

# ============================
# 10. Training Loop with Early Stopping
# ============================
for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    logging.info(f'\nEpoch {epoch+1}/{num_epochs}')
    print('-' * 10)
    logging.info('-' * 10)

    try:
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, all_preds, all_labels = validate_epoch(model, val_loader, criterion)
    except Exception as e:
        logging.error(f"Error during training/validation: {e}")
        print(f"Error during training/validation: {e}")
        sys.exit(1)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc.item())

    # Learning Rate Scheduler
    scheduler.step(val_loss)

    # Early Stopping
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        logging.info(f"Validation accuracy improved to {best_acc:.4f}")
        print(f"Validation accuracy improved to {best_acc:.4f}")
    else:
        epochs_no_improve += 1
        logging.info(f"No improvement in validation accuracy for {epochs_no_improve} epochs.")
        print(f"No improvement in validation accuracy for {epochs_no_improve} epochs.")
        if epochs_no_improve >= patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print()

# Load best model weights
model.load_state_dict(best_model_wts)
logging.info(f'Best Validation Accuracy: {best_acc:.4f}')
print(f'Best Validation Accuracy: {best_acc:.4f}')

# Save the model
model_path = os.path.join(results_dir, 'best_model_resnet18.pth')
torch.save(model.state_dict(), model_path)
logging.info(f"Model saved as '{model_path}'")
print(f"Model saved as '{model_path}'")

# ============================
# 11. Visualization of Training
# ============================
try:
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    training_curves_path = os.path.join(results_dir, 'training_curves_resnet18.png')
    plt.savefig(training_curves_path)
    plt.close()
    logging.info(f"Training curves saved as '{training_curves_path}'")
    print(f"Training curves saved as '{training_curves_path}'")
except Exception as e:
    logging.error(f"Error during training visualization: {e}")
    print(f"Error during training visualization: {e}")

# ============================
# 12. Final Evaluation
# ============================
# Final evaluation on the validation set
try:
    _, _, all_preds, all_labels = validate_epoch(model, val_loader, criterion)
except Exception as e:
    logging.error(f"Error during final evaluation: {e}")
    print(f"Error during final evaluation: {e}")

# Confusion Matrix
try:
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    confusion_matrix_path = os.path.join(results_dir, 'confusion_matrix_resnet18.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    logging.info(f"Confusion matrix saved as '{confusion_matrix_path}'")
    print(f"Confusion matrix saved as '{confusion_matrix_path}'")
except Exception as e:
    logging.error(f"Error during confusion matrix visualization: {e}")
    print(f"Error during confusion matrix visualization: {e}")

# Classification Report
try:
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    # Save classification report to a text file
    evaluation_results_path = os.path.join(results_dir, 'evaluation_results_resnet18.txt')
    with open(evaluation_results_path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names))
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))

    logging.info(f"Evaluation results saved to '{evaluation_results_path}'")
    print(f"Evaluation results saved to '{evaluation_results_path}'")
except Exception as e:
    logging.error(f"Error during classification report generation: {e}")
    print(f"Error during classification report generation: {e}")

# Print classification report
try:
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
except Exception as e:
    logging.error(f"Error printing classification report: {e}")
    print(f"Error printing classification report: {e}")

# ============================
# 13. Per-Class Metrics Visualization
# ============================
try:
    # Extract per-class metrics for visualization
    precision = [report[class_name]['precision'] for class_name in class_names]
    recall = [report[class_name]['recall'] for class_name in class_names]
    f1_score = [report[class_name]['f1-score'] for class_name in class_names]

    x = np.arange(len(class_names))  # label locations
    width = 0.2  # width of the bars

    plt.figure(figsize=(10,6))
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1_score, width, label='F1-Score')

    plt.ylabel('Scores')
    plt.xlabel('Classes')
    plt.title('Per-Class Precision, Recall, and F1-Score')
    plt.xticks(x, class_names)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    per_class_metrics_path = os.path.join(results_dir, 'per_class_metrics_resnet18.png')
    plt.savefig(per_class_metrics_path)
    plt.close()
    logging.info(f"Per-class metrics plot saved as '{per_class_metrics_path}'")
    print(f"Per-class metrics plot saved as '{per_class_metrics_path}'")
except Exception as e:
    logging.error(f"Error during per-class metrics visualization: {e}")
    print(f"Error during per-class metrics visualization: {e}")

# ============================
# 14. Additional Checks
# ============================
logging.info(f"\nUnique predictions: {np.unique(all_preds)}")
logging.info(f"Unique true labels: {np.unique(all_labels)}")
print("\nUnique predictions:", np.unique(all_preds))
print("Unique true labels:", np.unique(all_labels))

# ============================
# 15. GradCAM Visualization with Class Labels
# ============================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, input_image, target_class):
        self.model.eval()
        output = self.model(input_image)
        
        self.model.zero_grad()
        target = output[:, target_class]
        target.backward()

        # Pool the gradients across the spatial dimensions
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        # Weight the channels by corresponding gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        # Average the channels of the activations
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu(), 0)
        heatmap /= torch.max(heatmap) if torch.max(heatmap) != 0 else torch.tensor(1.0)
        return heatmap.numpy()

# Function to apply GradCAM and save the result with class label
def apply_gradcam(model, image, target_class, save_path, class_name):
    target_layer = model.layer4[-1]  # ResNet18's last convolutional layer
    grad_cam = GradCAM(model, target_layer)
    
    image_input = image.unsqueeze(0).to(device)
    heatmap = grad_cam.generate_heatmap(image_input, target_class)
    
    img = image.cpu().numpy().transpose(1, 2, 0)
    img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])  # Denormalize
    img = np.clip(img, 0, 1)
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * 0.4 + (img * 255).astype(np.uint8)
    superimposed_img = cv2.putText(superimposed_img, class_name, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Apply GradCAM to a few validation images
try:
    model.eval()
    processed_images = 0
    max_images = 5
    for i, (images, labels) in enumerate(val_loader):
        for j in range(images.size(0)):
            if processed_images >= max_images:
                break
            image = images[j].to(device)
            label = labels[j].item()
            class_name = class_names[label]
            
            save_path = os.path.join(results_dir, f'gradcam_resnet18_{processed_images}_{class_name}.png')
            apply_gradcam(model, image, label, save_path, class_name)
            
            processed_images += 1
        if processed_images >= max_images:
            break

    logging.info("GradCAM visualizations saved with class labels.")
    print("GradCAM visualizations saved with class labels.")
except Exception as e:
    logging.error(f"Error during GradCAM visualization: {e}")
    print(f"Error during GradCAM visualization: {e}")
