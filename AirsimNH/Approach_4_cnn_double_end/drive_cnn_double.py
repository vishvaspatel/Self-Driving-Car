import airsim
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
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

# Multi-task CNN Architecture (unchanged)
class MultiTaskCNN(nn.Module):
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
        
        self.shared_fc = nn.Sequential(
            nn.Linear(256*4*4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.steering_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        self.throttle_head = nn.Sequential(
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
        x = self.shared_fc(x)
        
        steering = self.steering_head(x).squeeze()
        throttle = self.throttle_head(x).squeeze()
        
        return steering, throttle

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# Initialize and load model
model = MultiTaskCNN().to(device)

try:
    # Load saved state dict (update the path)
    state_dict = torch.load("model_cnn_double.pth", map_location=device)
    model.load_state_dict(state_dict, strict=True)
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

# Control smoother
class ControlSmoother:
    def __init__(self, steering_alpha=0.3, throttle_alpha=0.5):
        self.steering_alpha = steering_alpha
        self.throttle_alpha = throttle_alpha
        self.prev_steering = 0.0
        self.prev_throttle = 0.5
    
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
MIN_THROTTLE = 0.2
MAX_THROTTLE = 1
# Set model to evaluation mode
model.eval()

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
            throttle = torch.sigmoid(throttle_pred).item()  # Apply sigmoid to throttle
            
            # Smooth and clip
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
            
            # Display image
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