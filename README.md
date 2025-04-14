# Lateral Control for Autonomous Vehicles

This project focuses on enhancing the lateral control of autonomous vehicles using deep learning techniques. It integrates **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)** for lane detection and steering/throttle prediction in simulated environments like **Udacity Simulator** and **AirSimNH**. Object detection using **YOLOv5** and a lane detection module further improve vehicle safety and control.

---

## 🔍 Objective

- Compare CNNs and ViTs for lateral control tasks.
- Use Udacity and AirSimNH simulators for data generation and model training.
- Implement data augmentation and balancing techniques.
- Build multi-task learning models to predict steering and throttle simultaneously.
- Integrate object detection and lane detection for improved driving safety.
- Deploy the trained model on a web application.

---
## 1. Overview of Simulators

### 1.1 Udacity Simulator

This simulator, often integrated with Udacity’s online courses, provides focused environments and scenarios designed for learning specific self-driving concepts and algorithms. It typically features simplified physics and sensor models, prioritizing pedagogical clarity and ease of use for educational purposes. The simulator often utilizes Python and libraries specific to the Udacity curriculum.

![udacity](https://github.com/user-attachments/assets/2f667bac-b5b2-412e-bb08-05def1ef8a20)


### 1.2 AirSimNH Simulator

Based on Microsoft AirSim, this simulator offers a more photorealistic and complex environment with detailed physics and support for a wider range of realistic sensors like LiDAR and radar. It allows for greater flexibility in creating custom scenarios and integrates well with industry-standard tools and frameworks such as ROS and Unreal Engine, making it suitable for more advanced research and development. AirSimNH enables the simulation of diverse and challenging real-world conditions.

![airsimnh](https://github.com/user-attachments/assets/034b02fb-d1ea-410a-9f60-146f7a2259cc)

---

## 📁 Datasets

### 📦 Udacity Dataset

**Table: Dataset Transformations**

| Transformation Step              | Samples Count |
|----------------------------------|---------------|
| Total initial data               | 4053          |
| Data removed for balancing       | 2590          |
| Remaining data after balancing   | 1463          |
| Training set after augmentation  | 3511          |
| Validation set after augmentation| 878           |

---

### 📦 AirSimNH Dataset

**Table: Steering Angle Distribution Before and After Balancing**

| Steering Angle | Initial Count | Balanced Count |
|----------------|---------------|----------------|
| -0.5           | 1124          | 1124           |
| 0.0            | 26012         | 15000          |
| 0.5            | 7571          | 7571           |

---

## 🧠 Approaches

---

### 🔹 Approach 1: CNN (Udacity)

**Table: CNN Model Architecture**

| Layer (Type)  | Output Shape       | Parameters |
|---------------|--------------------|------------|
| Conv2D        | (None, 31, 98, 24)  | 1,824      |
| Conv2D        | (None, 14, 47, 36)  | 21,636     |
| Conv2D        | (None, 5, 22, 48)   | 43,248     |
| Conv2D        | (None, 1, 18, 64)   | 76,864     |
| Flatten       | (None, 1152)        | 0          |
| Dense         | (None, 100)         | 115,300    |
| Dense         | (None, 50)          | 5,050      |
| Dense         | (None, 10)          | 510        |
| Dense         | (None, 1)           | 11         |
| **Total**     |                    | **264,443**|

---

### 🔹 Approach 2: Autoencoders (Udacity)

**Table: Autoencoder Architecture**

| Layer (Type)       | Output Shape      | Parameters |
|--------------------|-------------------|------------|
| InputLayer         | (None, 100, 320, 3)| 0          |
| Conv2D             | (None, 100, 320, 8)| 224        |
| MaxPooling2D       | (None, 50, 160, 8) | 0          |
| Conv2D             | (None, 50, 160, 8) | 584        |
| UpSampling2D       | (None, 100, 320, 8)| 0          |
| Conv2D             | (None, 100, 320, 3)| 219        |
| **Total**          |                   | **1,027**  |

---

### 🔹 Approach 3: CNN on AirSimNH (Steering Only)

**Table: Detailed SteeringCNN Architecture**

| Layer (Type)          | Output Shape       | Parameters |
|-----------------------|--------------------|------------|
| Conv2d-1              | [-1, 32, 64, 64]    | 2,432      |
| BatchNorm2d-2         | [-1, 32, 64, 64]    | 64         |
| ReLU-3                | [-1, 32, 64, 64]    | 0          |
| MaxPool2d-4           | [-1, 32, 32, 32]    | 0          |
| Conv2d-5              | [-1, 64, 32, 32]    | 18,496     |
| BatchNorm2d-6         | [-1, 64, 32, 32]    | 128        |
| ReLU-7                | [-1, 64, 32, 32]    | 0          |
| MaxPool2d-8           | [-1, 64, 16, 16]    | 0          |
| Conv2d-9              | [-1, 128, 16, 16]   | 73,856     |
| BatchNorm2d-10        | [-1, 128, 16, 16]   | 256        |
| ReLU-11               | [-1, 128, 16, 16]   | 0          |
| MaxPool2d-12          | [-1, 128, 8, 8]     | 0          |
| Conv2d-13             | [-1, 256, 8, 8]     | 295,168    |
| BatchNorm2d-14        | [-1, 256, 8, 8]     | 512        |
| ReLU-15               | [-1, 256, 8, 8]     | 0          |
| MaxPool2d-16          | [-1, 256, 4, 4]     | 0          |
| AdaptiveAvgPool2d-17  | [-1, 256, 4, 4]     | 0          |
| Linear-18             | [-1, 512]           | 2,097,664  |
| BatchNorm1d-19        | [-1, 512]           | 1,024      |
| ReLU-20               | [-1, 512]           | 0          |
| Dropout-21            | [-1, 512]           | 0          |
| Linear-22             | [-1, 256]           | 131,328    |
| BatchNorm1d-23        | [-1, 256]           | 512        |
| ReLU-24               | [-1, 256]           | 0          |
| Dropout-25            | [-1, 256]           | 0          |
| Linear-26             | [-1, 1]             | 257        |
| **Total Parameters**  |                    | **2,621,697** |
| **Trainable Params**  |                    | **2,621,697** |

### 🔹 Approach 4: CNN on AirSimNH (Steering + Throttle)

**Table: CNN Architecture for Steering and Throttle Prediction**

| Layer (Type)         | Output Shape      | Parameters |
|----------------------|-------------------|------------|
| Conv2d-1             | [-1, 32, 64, 64]   | 2,432      |
| BatchNorm2d-2        | [-1, 32, 64, 64]   | 64         |
| ReLU-3               | [-1, 32, 64, 64]   | 0          |
| MaxPool2d-4          | [-1, 32, 32, 32]   | 0          |
| Conv2d-5             | [-1, 64, 32, 32]   | 18,496     |
| BatchNorm2d-6        | [-1, 64, 32, 32]   | 128        |
| ReLU-7               | [-1, 64, 32, 32]   | 0          |
| MaxPool2d-8          | [-1, 64, 16, 16]   | 0          |
| Conv2d-9             | [-1, 128, 16, 16]  | 73,856     |
| BatchNorm2d-10       | [-1, 128, 16, 16]  | 256        |
| ReLU-11              | [-1, 128, 16, 16]  | 0          |
| MaxPool2d-12         | [-1, 128, 8, 8]    | 0          |
| Conv2d-13            | [-1, 256, 8, 8]    | 295,168    |
| BatchNorm2d-14       | [-1, 256, 8, 8]    | 512        |
| ReLU-15              | [-1, 256, 8, 8]    | 0          |
| MaxPool2d-16         | [-1, 256, 4, 4]    | 0          |
| AdaptiveAvgPool2d-17 | [-1, 256, 4, 4]    | 0          |
| Linear-18            | [-1, 512]          | 2,097,664  |
| BatchNorm1d-19       | [-1, 512]          | 1,024      |
| ReLU-20              | [-1, 512]          | 0          |
| Dropout-21           | [-1, 512]          | 0          |
| Linear-22            | [-1, 256]          | 131,328    |
| BatchNorm1d-23       | [-1, 256]          | 512        |
| ReLU-24              | [-1, 256]          | 0          |
| Dropout-25           | [-1, 256]          | 0          |
| Linear-26 (Steering) | [-1, 1]            | 257        |
| Linear-27 (Throttle) | [-1, 256]          | 131,328    |
| BatchNorm1d-28       | [-1, 256]          | 512        |
| ReLU-29              | [-1, 256]          | 0          |
| Dropout-30           | [-1, 256]          | 0          |
| Linear-31 (Throttle) | [-1, 1]            | 257        |
| **Total Parameters** |                   | **2,953,794** |
| **Trainable Params** |                   | **2,953,794** |
| **Non-trainable**    |                   | 0          |

### 🔹 Approach 5: ViT on AirSimNH (Steering Only)

**Table: ViT Architecture for Steering Prediction**

| Layer             | Input Shape | Output Shape | Parameters |
|------------------|-------------|--------------|------------|
| Patch Embedding  | (3,224,224) | (196,384)    | ~590K      |
| Transformer Blocks | (196,384) | (196,384)    | ~21M       |
| LayerNorm        | (384,)      | (384,)       | 768        |
| Linear            | (384,)     | (256,)       | 98,560     |
| Linear            | (256,)     | (128,)       | 32,896     |
| Linear            | (128,)     | (1,)         | 129        |
| **Total**         |             |              | **~21.73M**|

---

### 🔹 Approach 6: ViT on AirSimNH (Steering + Throttle)

**Table: ViT Architecture for Steering and Throttle**

| Layer Type         | Input Shape | Output Shape |
|--------------------|-------------|--------------|
| ViT Backbone       | (3, 224, 224)| (384,)       |
| Shared Linear      | (384,)       | (256,)       |
| Steering Head      | (256,)       | (1,)         |
| Throttle Head      | (256,)       | (1,)         |
| Sigmoid Activation | (1,)         | (1,)         |

---

## 🚘 Advanced Module: Object + Lane Detection

- **YOLOv5** for object detection (pre-trained on COCO).
- **Canny Edge Detection + Hough Lines** for lane marking.
- Integrated with the ViT model for enhanced decision-making.

---

## 🌐 Deployment

Try our web demo:  
👉 **[Streamlit App](https://autonomouscarapp-app-iur6pfwku8u5pjovsjmcab.streamlit.app/)**

---

## 📦 Repository

Codebase:  
👉 **[GitHub Repository](https://github.com/vishvaspatel/Lateral-Control-for-Autonomous-Vehicles-Self-Driving-Car-)**

---

## 🎥 Demo

🎬 **[Video Demonstration](https://drive.google.com/file/d/1vSDp0IiL0rcNPOlmwP-Dps-3KsKO_GIY/view?usp=sharing)**

---

## 👥 Contributors

| Name                  | Roll No.    | Contributions                              |
|-----------------------|-------------|---------------------------------------------|
| Udit Kandpal          | M24CSE027   | ViT model & AirSimNH data                  |
| Om Patel              | M24CSA019   | Autoencoders & ViT deployment              |
| Vishvaskumar Patel    | M24CSE029   | CNN (AirSimNH) & Report writing           |
| Rahul Maurya          | M24CSA025   | CNN (Udacity) & dataset creation           |
| Patil Divya Kailash   | M24CSE018   | CNN (Udacity) & documentation              |

---

