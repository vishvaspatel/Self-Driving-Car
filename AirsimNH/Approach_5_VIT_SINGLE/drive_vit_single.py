import airsim
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import time

# Connect to AirSim
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)

# Initialize car controls
car_controls = airsim.CarControls()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CORRECT Model architecture that matches your saved checkpoint
class ViTSteeringModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Using vit_small_patch16_224 which has 384-dim features
        self.backbone = timm.create_model('vit_small_patch16_224', 
                                        pretrained=False,
                                        num_classes=0)
        
        # Head architecture that matches your saved model
        self.head = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze()

# Initialize and load model
model = ViTSteeringModel().to(device)

try:
    # Load saved state dict
    state_dict = torch.load("model_vit_single.pth", map_location=device)
    
    # Handle any potential naming mismatches
    if any(k.startswith('vit.') for k in state_dict.keys()):
        state_dict = {k.replace('vit.', 'backbone.'): v for k, v in state_dict.items()}
    
    # Load with strict=False to ignore unexpected keys
    model.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully!")
    
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Steering smoother
class SteeringSmoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.prev_steering = 0.0
    
    def smooth(self, new_steering):
        smoothed = self.alpha * new_steering + (1 - self.alpha) * self.prev_steering
        self.prev_steering = smoothed
        return smoothed

steering_smoother = SteeringSmoother()

# Control parameters
MAX_STEERING = 1.0
MIN_STEERING = -1.0
BASE_THROTTLE = 0.5
STEERING_THROTTLE_REDUCTION = 0.2

try:
    while True:
        start_time = time.time()
        
        # Get camera image
        try:
            responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
            if not responses or not responses[0].image_data_uint8:
                print("No image received, skipping frame")
                continue
                
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"Error getting image: {e}")
            continue

        # Preprocess and predict
        try:
            input_tensor = transform(img_rgb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                steering_pred = model(input_tensor).item()
                
            # Process prediction
            steering = steering_smoother.smooth(steering_pred)
            steering = np.clip(steering, MIN_STEERING, MAX_STEERING)
            throttle = BASE_THROTTLE * (1 - abs(steering) * STEERING_THROTTLE_REDUCTION)
            
            # Set controls
            car_controls.throttle = throttle
            car_controls.steering = steering
            car_controls.brake = 0.0
            client.setCarControls(car_controls)
            
            # Debug output
            fps = 1.0 / (time.time() - start_time)
            print(f"Steering: {steering:.3f} | Throttle: {throttle:.3f} | FPS: {fps:.1f}")
            
        except Exception as e:
            print(f"Prediction error: {e}")
            continue
            
except KeyboardInterrupt:
    print("\nStopping vehicle...")
    car_controls.throttle = 0.0
    car_controls.brake = 1.0
    client.setCarControls(car_controls)
    client.enableApiControl(False)
    print("Disconnected from AirSim")