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

# Multi-task model architecture (exactly matching your training code)
class MultiTaskViT(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared backbone
        self.backbone = timm.create_model(
            'vit_small_patch16_224',  # Assuming this matches your Config.MODEL_NAME
            pretrained=False,
            num_classes=0
        )
        
        # Shared layers
        self.shared_head = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # Task-specific heads
        self.steering_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        self.throttle_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_head(features)
        
        steering = self.steering_head(shared).squeeze()
        throttle = torch.sigmoid(self.throttle_head(shared)).squeeze()  # Added sigmoid for throttle
        
        return steering, throttle

# Initialize and load model
model = MultiTaskViT().to(device)

try:
    # Load saved state dict
    state_dict = torch.load("final_model.pth", map_location=device)
    
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

# Control smoother for both steering and throttle
class ControlSmoother:
    def __init__(self, steering_alpha=0.3, throttle_alpha=0.5):
        self.steering_alpha = steering_alpha
        self.throttle_alpha = throttle_alpha
        self.prev_steering = 0.0
        self.prev_throttle = 0.5  # Default value
    
    def smooth(self, new_steering, new_throttle):
        smoothed_steering = self.steering_alpha * new_steering + (1 - self.steering_alpha) * self.prev_steering
        smoothed_throttle = self.throttle_alpha * new_throttle + (1 - self.throttle_alpha) * self.prev_throttle
        self.prev_steering = smoothed_steering
        self.prev_throttle = smoothed_throttle
        return smoothed_steering, smoothed_throttle

control_smoother = ControlSmoother()

# Control parameters
MAX_STEERING = 1.0
MIN_STEERING = -1.0
MIN_THROTTLE = 0.2  # Minimum throttle to keep the car moving
MAX_THROTTLE = 1.0

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
                steering_pred, throttle_pred = model(input_tensor)
                
            # Process predictions
            steering = steering_pred.item()
            throttle = throttle_pred.item()
            
            # Smooth and clip the control values
            steering, throttle = control_smoother.smooth(steering, throttle)
            steering = np.clip(steering, MIN_STEERING, MAX_STEERING)
            throttle = np.clip(throttle, MIN_THROTTLE, MAX_THROTTLE)
            
            # Set controls
            car_controls.throttle = throttle
            car_controls.steering = steering
            car_controls.brake = 0.0
            client.setCarControls(car_controls)
            
            # Debug output
            fps = 1.0 / (time.time() - start_time)
            print(f"Steering: {steering:.3f} | Throttle: {throttle:.3f} | FPS: {fps:.1f}")
            
            # Optional: Display the image
            cv2.imshow('AirSim View', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Prediction error: {e}")
            continue
            
except KeyboardInterrupt:
    print("\nStopping vehicle...")
    car_controls.throttle = 0.0
    car_controls.brake = 1.0
    client.setCarControls(car_controls)
    client.enableApiControl(False)
    cv2.destroyAllWindows()
    print("Disconnected from AirSim")