# Lateral Control for Autonomous Vehicles: Project Report

**Date:** May 2025
**Institution:** Indian Institute of Technology Jodhpur
![airsimnh](https://github.com/user-attachments/assets/78a0ef64-e284-4ff2-9f61-ee31ca161197)
![udacity](https://github.com/user-attachments/assets/b5bd59e1-3af6-41b1-91f5-8ebbda16adcb)



---

## Abstract

This project presents a deep-learning–based pipeline for robust lateral control of autonomous vehicles. We compare Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) for end-to-end steering and throttle prediction, integrating object and lane detection modules for enhanced safety. Synthetic datasets are generated in Udacity and AirSimNH simulators, balanced and augmented to train multi-task models. A web-based Streamlit app demonstrates real-time inference.

**Keywords:** lateral control, CNN, Vision Transformer, YOLOv5, simulator, multi-task learning

---

## 1. Introduction

Autonomous vehicles require precise lateral control to maintain lane discipline and ensure passenger safety. Traditional control methods rely on hand-crafted features and decoupled perception–planning pipelines. This work explores end-to-end deep learning approaches, leveraging modern architectures (CNNs, ViTs) and real-time perception modules (object and lane detection) to improve robustness in simulation.

## 2. Project Objectives

1. Evaluate CNN and ViT models for steering-only and steering+throttle tasks.
2. Generate and preprocess synthetic driving data using Udacity and AirSimNH simulators.
3. Implement histogram-based balancing and data augmentation.
4. Integrate YOLOv5 for obstacle detection and a Canny+Hough pipeline for lane marking.
5. Develop emergency brake and adaptive steering systems.
6. Deploy the trained pipeline in a Streamlit web application.

---

## 3. Experimental Setup

### 3.1 Simulators

* **Udacity Simulator:** Simplified physics, designed for educational experiments with center/left/right camera streams.
* **AirSimNH Simulator:** Photorealistic environment with detailed sensor models (LiDAR, radar) and ROS/Unreal Engine integration.

### 3.2 Dataset Preparation

* **Udacity Dataset:** 4,053 raw samples; after histogram balancing (removing overrepresented steering angles) 1,463 samples remain; post-augmentation yields 4,389 training and validation images.
* **AirSimNH Dataset:** Initial distribution heavily centered at zero steering; balanced via bin capping (15,000 zero-angle samples) and augmentation to maintain class parity.

Data augmentation techniques included random zoom, pan, brightness adjustment, horizontal flips, and synthetic noise injection.

---

## 4. Methodology

### 4.1 Perception Modules

* **Object Detection (YOLOv5):** Pre-trained on COCO; fine-tuned on simulator images to detect vehicles, pedestrians, and obstacles.
* **Lane Detection:** Canny edge extraction followed by Hough line transformation to locate lane boundaries.

### 4.2 Model Architectures

* **CNN Models:** Sequential conv–pool blocks with fully connected regression heads. Architectures evaluated:

  * CNN-Udacity (steering only) — ∼264K parameters.
  * CNN-AirSimNH (steering only) — ∼2.62M parameters.
  * CNN-AirSimNH (steering + throttle) — ∼2.95M parameters.

* **Autoencoder Baseline:** CNN autoencoder for feature extraction, ∼1K parameters.

* **Vision Transformers (ViT):** Patch embedding backbone with transformer encoder layers; ∼21.7M parameters for steering-only, extended heads for throttle.

### 4.3 Multi-Task Learning

Shared backbone with separate regression heads for steering and throttle; custom loss weighting to balance tasks based on validation error scales.

### 4.4 Safety and Smoothing

* Exponential moving average on control outputs to reduce jitter.
* Emergency braking triggers when YOLO detects obstacle within predefined proximity zone.
* Throttle reduction during high-curvature scenarios to maintain vehicle stability.

---

## 5. Results and Discussion

Performance metrics include Mean Absolute Error (MAE) for steering and throttle.

| Model                     | Steering MAE | Throttle MAE |
| ------------------------- | -----------: | -----------: |
| CNN-Udacity               |        0.025 |            — |
| CNN-AirSimNH (steer only) |        0.018 |            — |
| CNN-AirSimNH (multi-task) |        0.020 |        0.032 |
| ViT-AirSimNH (steer only) |        0.015 |            — |
| ViT-AirSimNH (multi-task) |        0.017 |        0.030 |

Visualizations and training curves are available in the project report (PDF).

---

## 6. Deployment

* **Streamlit App:** Live demo hosted [here](https://autonomouscarapp-app-iur6pfwku8u5pjovsjmcab.streamlit.app/).
* **Video Walkthrough:** [YouTube](https://drive.google.com/file/d/1vSDp0IiL0rcNPOlmwP-Dps-3KsKO_GIY/view)

Example usage:

```bash
streamlit run app/app.py
```

---

## 7. Conclusion and Future Work

This study demonstrates the efficacy of deep learning for lateral control with integrated perception and safety modules. ViT-based models achieved the lowest MAE and collision rates, at the expense of higher parameter counts. Future directions include real-world hardware-in-the-loop testing, reinforcement learning for closed-loop adaptation, and sensor fusion with LiDAR.

---

## References

1. Redmon, J. et al. "YOLOv5: Real-Time Object Detection." *arXiv preprint*.
2. Dosovitskiy, A. et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2021*.
3. Code repository: [https://github.com/vishvaspatel/Lateral-Control-for-Autonomous-Vehicles-Self-Driving-Car-](https://github.com/vishvaspatel/Lateral-Control-for-Autonomous-Vehicles-Self-Driving-Car-)

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
