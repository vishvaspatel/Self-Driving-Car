import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

print("Starting training...")
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_Model_2_single.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Paths (same as your original)
    CSV_PATH = '/csehome/m24cse029/test/balanced_steering_dataset.csv'
    IMAGE_DIR = '/csehome/m24cse029/test/2025-04-09-23-44-45/images'
    SAVE_DIR = '/csehome/m24cse029/test/Models_2_single'
    
    # Training
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.01
    
    # Model
    IMAGE_SIZE = 224
    
    # System
    NUM_WORKERS = 4
    PIN_MEMORY = True
    SEED = 42

# Set random seed
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Dataset class (same as original)
class SteeringDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = self.data['ImageFile'].values
        self.steering = self.data['Steering'].values
        logger.info(f"Loaded dataset with {len(self)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {str(e)}")
            image = Image.new('RGB', (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            steering = np.mean(self.steering)
            return self._apply_transform(image), torch.tensor(steering, dtype=torch.float32)

        steering = self.steering[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(steering, dtype=torch.float32)

    def _apply_transform(self, image):
        if self.transform:
            return self.transform(image)
        return transforms.ToTensor()(image)

# CNN Architecture
class SteeringCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature extraction backbone
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Adaptive pooling to handle varying spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(256*4*4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.regressor(x).squeeze()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# Transforms (same as original)
train_transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    # Create directories
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    # Load datasets
    full_dataset = SteeringDataset(
        csv_path=Config.CSV_PATH,
        img_dir=Config.IMAGE_DIR,
        transform=train_transform
    )
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Apply val transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    # Initialize model
    model = SteeringCNN().to(device)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training components
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(Config.EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]")
        
        for images, steering in train_bar:
            images = images.to(device, non_blocking=True)
            steering = steering.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, steering)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            train_bar.set_postfix(loss=loss.item())
        
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_loss = 0.0
        model.eval()
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]")
        
        with torch.no_grad():
            for images, steering in val_bar:
                images = images.to(device, non_blocking=True)
                steering = steering.to(device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, steering)
                
                val_loss += loss.item() * images.size(0)
                val_bar.set_postfix(loss=loss.item())
        
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        scheduler.step()
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        
        # Save best model
        model_path = os.path.join(Config.SAVE_DIR, f'model_2_single_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved new best model to {model_path}")
    
    # Save final model and training history
    final_model_path = os.path.join(Config.SAVE_DIR, 'model_2_single_final.pth')
    torch.save(model.state_dict(), final_model_path)
    
    history_path = os.path.join(Config.SAVE_DIR, 'modle_2_single_training_history.npy')
    np.save(history_path, history)
    
    logger.info(f"Training complete. Final model saved to {final_model_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CNN Training and Validation Loss')
    plt.legend()
    plot_path = os.path.join(Config.SAVE_DIR, 'cnn_loss_curve.png')
    plt.savefig(plot_path)
    logger.info(f"Loss curve saved to {plot_path}")

if __name__ == "__main__":
    try:
        print("Starting training...")
        main()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise