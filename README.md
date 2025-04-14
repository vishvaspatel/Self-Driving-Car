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

## 🧪 Simulators Used

| Simulator       | Description |
|-----------------|-------------|
| **Udacity**     | Simple simulator for educational use with basic physics and 3-camera setup. |
| **AirSimNH**    | Advanced, photorealistic simulator from Microsoft with support for complex physics and sensors. |

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

| Layer (Type)         | Output Shape     | Parameters |
|----------------------|------------------|------------|
| Conv2d               | [-1, 32, 64, 64]  | 2,432      |
| BatchNorm2d          | [-1, 32, 64, 64]  | 64         |
| Conv2d               | [-1, 64, 32, 32]  | 18,496     |
| BatchNorm2d          | [-1, 64, 32, 32]  | 128        |
| Conv2d               | [-1, 128, 16, 16] | 73,856     |
| BatchNorm2d          | [-1, 128, 16, 16] | 256        |
| Conv2d               | [-1, 256, 8, 8]   | 295,168    |
| BatchNorm2d          | [-1, 256, 8, 8]   | 512        |
| Linear               | [-1, 512]         | 2,097,664  |
| BatchNorm1d          | [-1, 512]         | 1,024      |
| Linear               | [-1, 256]         | 131,328    |
| Linear               | [-1, 1]           | 257        |
| **Total**            |                  | **2,621,697** |

---

### 🔹 Approach 4: CNN on AirSimNH (Steering + Throttle)

**Table: CNN Architecture for Steering and Throttle Prediction**

| Layer (Type)         | Output Shape     | Parameters |
|----------------------|------------------|------------|
| Conv2d               | [-1, 32, 64, 64]  | 2,432      |
| Conv2d               | ...              | ...        |
| Linear (Steering)    | [-1, 1]           | 257        |
| Linear (Throttle)    | [-1, 256]         | 131,328    |
| Linear (Throttle)    | [-1, 1]           | 257        |
| **Total**            |                  | **2,953,794** |

---

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
👉 **[Streamlit App](https://requirementstxt-vhoxy8j2hvuzsqcfbesfxj.streamlit.app/)**

---

## 📦 Repository

Codebase:  
👉 **[GitHub Repository](https://github.com/vishvaspatel/Lateral-Control-for-Autonomous-Vehicles-Self-Driving-Car-)**

---

## 🎥 Demo

🎬 **[Video Demonstration](https://drive.google.com/file/d/1KY-AiZtJS-uPN8IbmP-pbl9t1usOdNmM/view?usp=drive_link)**

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

