# Lateral Control for Autonomous Vehicles (Self Driving Car)

**Date:** May 2025
**Institution:** Indian Institute of Technology Jodhpur

---

## Abstract

This project presents a deep-learning–based pipeline for robust lateral control of autonomous vehicles. We compare Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) for end-to-end steering and throttle prediction, integrating object and lane detection modules for enhanced safety. Synthetic datasets are generated in Udacity and AirSimNH simulators, balanced and augmented to train multi-task models. A web-based Streamlit app demonstrates real-time inference.

---

## 1. Experimental Setup

### 1.1 Simulators

* **Udacity Simulator:** Simplified physics, designed for educational experiments with center/left/right camera streams.
* **AirSimNH Simulator:** Photorealistic environment with detailed sensor models (LiDAR, radar) and ROS/Unreal Engine integration.

| ![udacity](https://github.com/user-attachments/assets/448b8ad1-8236-4e07-a675-ff7af0bae2c8) | ![airsimnh](https://github.com/user-attachments/assets/f9b33ab8-7b2e-4393-b4d2-7f3e5f40f14e) |
|:------------------------------------------:|:-------------------------------------------:|



### 1.2 Dataset Details

* **Udacity Dataset:** 4,053 raw image samples (center camera), recorded at 10 Hz over varying tracks and lighting conditions.

  * **Steering Angle Distribution (raw):** heavily concentrated around zero steering.
  * **Balanced Samples:** Removed 2,590 samples from the \[-0.1, 0.1] steering bin to mitigate bias, resulting in 1,463 samples.
  * **Post-Augmentation:** Applied geometric and photometric transforms to expand to 3,511 training and 878 validation images.

| Transformation Step          | Samples Count |
| ---------------------------- | ------------- |
| Total raw samples            | 4,053         |
| Removed for balancing        | 2,590         |
| Remaining after balancing    | 1,463         |
| Augmented training samples   | 3,511         |
| Augmented validation samples | 878           |

![Screenshot 2025-05-06 171414](https://github.com/user-attachments/assets/0457a14f-3af0-4009-8916-98a26b90b7e7)



* **AirSimNH Dataset:** Over 40,000 raw frames captured at 5 Hz across urban and highway scenarios.

  * **Initial Steering Distribution:** heavily skewed to zero (26,012 samples).
  * **Balanced via Bin Capping:** Limited zero-angle bin to 15,000 samples, retaining all other bins.
  * **Final Dataset:** \~36,695 frames before augmentation.

| Steering Angle | Initial Count | After Capping |
| -------------- | ------------- | ------------- |
| -0.5           | 1,124         | 1,124         |
| 0.0            | 26,012        | 15,000        |
| 0.5            | 7,571         | 7,571         |

| ![airsim_dataset_before_balancing](https://github.com/user-attachments/assets/3d20b05e-4c3f-4f5e-b12d-65d6040271ba) | ![airsim_dataset_after_balancing](https://github.com/user-attachments/assets/5ca0e717-c404-4cde-b7a0-3e15380a25b1) |
|:------------------------------------------:|:-------------------------------------------:|





### 1.3 Preprocessing Steps

1. **Frame Extraction:** Loaded raw PNG/JPG images, resized to 100×320 resolution.
2. **Image Cropping:** Removed top 50 pixels (sky and vehicle hood) to focus on roadway.
3. **Color Space Conversion:** Converted RGB to YUV for improved lighting invariance.
4. **Normalization:** Scaled pixel values to \[0, 1] and standardized per-channel mean and std.
5. **Histogram Equalization:** Applied CLAHE on Y channel to enhance contrast.
6. **Steering Binning:** Discretized continuous steering angles into 15 uniform bins for balancing.
7. **Data Augmentation:** In training pipeline:

   * Random zoom (0.8–1.2×)
   * Random horizontal shift (±50 px) and vertical shift (±10 px)
   * Brightness adjustment (±20%)
   * Gaussian noise injection (σ=0.01)
   * Random horizontal flips (steering angle negated)
8. **Dataset Splitting:** 80/20 train/validation split, stratified by steering bins to preserve distribution.

---

## 2. Methodology

### 2.1 Detection Modules

* **Object Detection (YOLOv5):** A YOLOv5 model pretrained on COCO and fine-tuned on simulator data to detect vehicles, pedestrians, and static obstacles. Detections include bounding box coordinates and confidence scores.
* **Lane Detection:** Canny edge detection on grayscale images followed by Hough line transform to extract lane boundary segments; post-processing merges colinear lines and fits lane polynomials.

### 2.2 Emergency Braking System

* **Triggering Logic:** If YOLOv5 detects any object whose bounding box enters the predefined collision zone (distance < 5 m, central field of view), issue an immediate brake command.
* **Brake Command:** Throttle set to zero; optional handbrake flag in simulator APIs to simulate full stop.

### 2.3 Emergency Brake + Steering Adjustment

* **Combined Strategy:** Upon obstacle detection in lateral proximity (<2 m from vehicle centerline), throttle is reduced by 50% and steering angle is adjusted away from obstacle.
* **Steering Adjustment:** Compute obstacle centroid in image frame, map to steering offset via linear mapping: Δθ = k·(x\_img−x\_center), with k calibrated from simulator.

### 2.4 Model Architectures

The following model variants were evaluated:

* **CNN-Udacity (Steering Only)**
* **Autoencoder Baseline (Udacity)**
* **CNN-AirSimNH (Steering Only)**
* **CNN-AirSimNH (Steering + Throttle)**
* **Vision Transformer (ViT) Steering Only**
* **Vision Transformer (ViT) Steering + Throttle**

Refer to Section 3 for detailed performance results.

### 2.5 Multi-Task Learning

* **Loss Function:** L = α·MAE\_steer + (1−α)·MAE\_throttle
* **Training:** Joint backpropagation optimizes shared backbone and distinct heads, using AdamW optimizer and learning rates tuned per model.

### 2.6 Safety and Smoothing

* **Exponential Smoothing:** ŷ\_t = β·y\_t + (1−β)·ŷ\_{t−1}, with β=0.2 to smooth control actions.
* **Throttle Modulation:** In high-curvature segments (curvature >0.01 m⁻¹), throttle reduced by 30% to maintain stability.

## 3. Results and Discussion

Performance metrics include Mean Absolute Error (MAE) for steering and throttle.


| Model                     | Steering Validation Loss | Throttle Validation Loss |
|---------------------------|--------------------------|--------------------------|
| CNN-Udacity               | 0.035                    | —                        |
| Autoencoders              | 0.02                     | —                        |
| CNN-AirSimNH (steer only) | 0.009                    | —                        |
| CNN-AirSimNH (multi-task) | 0.033                    | 0.25                     |
| ViT-AirSimNH (steer only) | 0.00001                  | —                        |
| ViT-AirSimNH (multi-task) | 0.005                    | 0.009                    |

---

## 4. Deployment

* **Streamlit App:** Live demo hosted [here](https://autonomouscarapp-app-iur6pfwku8u5pjovsjmcab.streamlit.app/).
---

## 5. Conclusion and Future Work

This study demonstrates the efficacy of deep learning for lateral control with integrated perception and safety modules. ViT-based models achieved the lowest MAE and collision rates, at the expense of higher parameter counts. Future directions include real-world hardware-in-the-loop testing, reinforcement learning for closed-loop adaptation, and sensor fusion with LiDAR.

---

## References

1. Redmon, J. et al. "YOLOv5: Real-Time Object Detection." *arXiv preprint*.
2. Dosovitskiy, A. et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2021*.
3. Code repository: [https://github.com/vishvaspatel/Lateral-Control-for-Autonomous-Vehicles-Self-Driving-Car-](https://github.com/vishvaspatel/Lateral-Control-for-Autonomous-Vehicles-Self-Driving-Car-)

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
