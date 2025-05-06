import airsim
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from torchvision import transforms
import time
import warnings
import csv
import os
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)

# Connect to AirSim
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)

# Initialize car controls
car_controls = airsim.CarControls()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load YOLOv5 for object detection
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
yolo_model.conf = 0.5  # Confidence threshold
yolo_model.classes = [2, 5, 7]  # Filter for cars, buses, trucks only

# Vision Transformer Steering Model
class ViTSteeringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch16_224', 
                                        pretrained=False,
                                        num_classes=0)
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

# Initialize and load steering model
model = ViTSteeringModel().to(device)
try:
    state_dict = torch.load("Approach_5_VIT_SINGLE\model_vit_single.pth", map_location=device)
    if any(k.startswith('vit.') for k in state_dict.keys()):
        state_dict = {k.replace('vit.', 'backbone.'): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    print("Steering model loaded successfully!")
except Exception as e:
    print(f"Error loading steering model: {e}")
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
BASE_THROTTLE = 0.3
STEERING_THROTTLE_REDUCTION = 0.2
EMERGENCY_BRAKE_DISTANCE = 30  # pixels threshold for braking
COLLISION_DISTANCE = 60  # pixels threshold for steering adjustment

# Setup CSV logging
log_dir = "steering_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(log_dir, f"steering_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
csv_file = open(log_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'timestamp', 
    'raw_steering', 
    'smoothed_steering', 
    'adjusted_steering', 
    'throttle', 
    'brake', 
    'emergency_brake', 
    'adjustment_due_to_object', 
    'fps'
])

def detect_obstacles(image):
    """Detect cars and other obstacles using YOLOv5"""
    results = yolo_model(image)
    detections = []
    
    for *xyxy, conf, cls in results.xyxy[0]:
        if conf > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = map(int, xyxy)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'size': (width, height),
                'class': results.names[int(cls)],
                'confidence': float(conf)
            })
    
    return detections

def calculate_avoidance(steering, detections, frame_width, frame_height):
    """Adjust steering only if collision is imminent"""
    if not detections:
        return steering, False, False
    
    emergency_brake = False
    adjusted_steering = steering
    adjustment_due_to_object = False
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        center_x, center_y = det['center']
        
        # Only consider objects in our immediate path (bottom center of image)
        path_left = frame_width//2 - frame_width//4
        path_right = frame_width//2 + frame_width//4
        path_bottom = frame_height - COLLISION_DISTANCE
        
        if (x1 < path_right and x2 > path_left and y2 > path_bottom):
            # Object is in our immediate path
            adjustment_due_to_object = True
            if center_x > frame_width//2:  # Object on right, steer left
                adjusted_steering = max(MIN_STEERING, steering - 0.5)
            else:  # Object on left, steer right
                adjusted_steering = min(MAX_STEERING, steering + 0.5)
            
            # Emergency brake if very close
            if y2 > frame_height - EMERGENCY_BRAKE_DISTANCE:
                emergency_brake = True
    
    return adjusted_steering, emergency_brake, adjustment_due_to_object

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
            
            # For visualization
            display_img = img_rgb.copy()
            frame_height, frame_width = img_rgb.shape[:2]
            
        except Exception as e:
            print(f"Error getting image: {e}")
            continue

        # Object detection
        detections = detect_obstacles(img_rgb)
        
        # Visualize detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(display_img, f"{det['class']} {det['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Preprocess and predict steering
        try:
            input_tensor = transform(img_rgb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                steering_pred = model(input_tensor).item()
                
            # Process prediction
            smoothed_steering = steering_smoother.smooth(steering_pred)
            smoothed_steering = np.clip(smoothed_steering, MIN_STEERING, MAX_STEERING)
            
            # Only adjust steering if collision is imminent
            adjusted_steering, emergency_brake, adjustment_due_to_object = calculate_avoidance(smoothed_steering, detections, frame_width, frame_height)
            
            # Calculate throttle
            throttle = BASE_THROTTLE * (1 - abs(adjusted_steering) * STEERING_THROTTLE_REDUCTION)
            if emergency_brake:
                throttle = 0.0
                brake = 1.0
            else:
                brake = 0.0
            
            # Set controls
            car_controls.throttle = throttle
            car_controls.steering = adjusted_steering
            car_controls.brake = brake
            client.setCarControls(car_controls)
            
            # Log steering data
            fps = 1.0 / (time.time() - start_time)
            csv_writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                f"{steering_pred:.3f}",
                f"{smoothed_steering:.3f}",
                f"{adjusted_steering:.3f}",
                f"{throttle:.3f}",
                f"{brake:.1f}",
                str(emergency_brake),
                str(adjustment_due_to_object),
                f"{fps:.1f}"
            ])
            csv_file.flush()  # Ensure data is written immediately
            
            # Display
            cv2.imshow('Object Detection', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Debug output
            print(f"Steering: {adjusted_steering:.3f} | Throttle: {throttle:.3f} | Brake: {brake:.1f} | FPS: {fps:.1f}")
            
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
    csv_file.close()  # Close the CSV file
    print(f"Steering data saved to {log_filename}")
    print("Disconnected from AirSim")