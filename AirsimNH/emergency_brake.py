import airsim
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from torchvision import transforms
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Fixed capitalization

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
EMERGENCY_BRAKE_DISTANCE = 25 # pixels threshold for braking
COLLISION_DISTANCE = 60  

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
    """Check for emergency brake condition without adjusting steering"""
    if not detections:
        return steering, False
    
    emergency_brake = False
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        
        # Only consider objects in our immediate path (bottom center of image)
        path_left = frame_width//2 - frame_width//4
        path_right = frame_width//2 + frame_width//4
        path_bottom = frame_height - COLLISION_DISTANCE
        
        if (x1 < path_right and x2 > path_left and y2 > path_bottom):
            # Emergency brake if very close
            if y2 > frame_height - EMERGENCY_BRAKE_DISTANCE:
                emergency_brake = True
    
    return steering, emergency_brake

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
            steering = steering_smoother.smooth(steering_pred)
            steering = np.clip(steering, MIN_STEERING, MAX_STEERING)
            
            # Check for emergency brake without adjusting steering
            steering, emergency_brake = calculate_avoidance(steering, detections, frame_width, frame_height)
            
            # Calculate throttle
            throttle = BASE_THROTTLE * (1 - abs(steering) * STEERING_THROTTLE_REDUCTION)
            if emergency_brake:
                throttle = 0.0
                brake = 1.0
            else:
                brake = 0.0
            
            # Set controls
            car_controls.throttle = throttle
            car_controls.steering = steering
            car_controls.brake = brake
            client.setCarControls(car_controls)
            
            # Display
            cv2.imshow('Object Detection', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Debug output
            fps = 1.0 / (time.time() - start_time)
            print(f"Steering: {steering:.3f} | Throttle: {throttle:.3f} | Brake: {brake:.1f} | FPS: {fps:.1f}")
            
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