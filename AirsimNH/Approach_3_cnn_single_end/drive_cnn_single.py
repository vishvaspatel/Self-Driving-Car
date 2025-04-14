import airsim
import cv2
import numpy as np
import torch
import torch.nn as nn
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

# Model architecture
class SteeringCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
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

# Initialize and load model
model = SteeringCNN().to(device)

try:
    state_dict = torch.load("cnn_model_single.pth", map_location=device)
    model.load_state_dict(state_dict, strict=True)
    print("Model loaded successfully!")
    model.eval()  # Set to evaluation mode
    
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Image preprocessing
from torchvision import transforms
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