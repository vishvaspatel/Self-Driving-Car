# Lateral Control for Autonomous Vehicles

[![Streamlit Demo](https://img.shields.io/badge/Streamlit-Demo-blue)](https://autonomouscarapp-app-iur6pfwku8u5pjovsjmcab.streamlit.app/)  [![GitHub Repo](https://img.shields.io/badge/GitHub-Source-black)](https://github.com/vishvaspatel/Lateral-Control-for-Autonomous-Vehicles-Self-Driving-Car-)

A deep-learning pipeline for robust lateral control of autonomous vehicles, combining lane/object perception with steering & throttle prediction in simulated environments.

---

## 🚀 Features

* **Lateral Control**: End-to-end steering & throttle prediction using CNNs and Vision Transformers (ViTs).
* **Perception Modules**:

  * **Lane Detection**: Canny + Hough transform for real-time lane marking.
  * **Object Detection**: YOLOv5 for obstacle identification (cars, buses, pedestrians).
* **Emergency Systems**:

  * **E‑Brake**: YOLO‑driven emergency braking when obstacles detected in collision zone.
  * **Adaptive Steering**: Dynamic steering adjustments to avoid detected obstacles.
* **Data Pipelines**:

  * Synthetic data from **Udacity** & **AirSimNH** simulators.
  * Balancing (histogram binning) & augmentation (zoom, pan, brightness, flip).
* **Multi‑Task Learning**: Simultaneous steering & throttle control with custom loss weighting.
* **Smoothing & Safety**: Exponential smoothing of steering, throttle reduction during sharp turns.
* **Web Demo**: Interactive Streamlit app for live inference.

---

## 📂 Repository Structure

```bash
├── data/                     # Raw and preprocessed datasets (Udacity, AirSimNH)
├── models/                   # Trained CNN & ViT checkpoints
├── notebooks/                # Jupyter notebooks for EDA & training
├── src/                      # Core code: data loaders, architectures, training loops
│   ├── architectures/        # CNN, Autoencoder, ViT definitions
│   ├── detection/            # YOLOv5 integration & lane detection scripts
│   ├── inference/            # Controllers: lateral, e‑brake, steering adjust
│   └── utils/                # Preprocessing & augmentation pipelines
├── app/                      # Streamlit web application
├── report/                   # Full project report (PDF)
└── README.md                 # This file
```

---

## ⚙️ Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/vishvaspatel/Lateral-Control-for-Autonomous-Vehicles-Self-Driving-Car- .
   cd Lateral-Control-for-Autonomous-Vehicles-Self-Driving-Car-
   ```

2. **Create environment & install**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Download Datasets**

   * Udacity: Place center/left/right images and CSV under `data/udacity/`
   * AirSimNH: Place `.png` images and driving logs under `data/airsimnh/`

4. **Prepare Models**

   ```bash
   python src/utils/preprocess.py  # balancing & augmentation
   python src/train.py --config configs/cnn_udacity.yaml
   ```

---

## 🏗️ Usage

* **Train**:

  ```bash
  python src/train.py --config configs/vit_airsim_st.yaml  # ViT steering-only
  python src/train.py --config configs/cnn_multi_task.yaml  # CNN steering+throttle
  ```

* **Evaluate**:

  ```bash
  python src/evaluate.py --model-path models/cnn_multi_task.pt --dataset airsimnh
  ```

* **Run Web Demo**:

  ```bash
  cd app
  streamlit run app.py
  ```

---

## 📊 Results & Visualizations

Refer to the [project report (PDF)](report/CV_Project_Report.pdf) for detailed plots, error curves, and performance comparisons between CNN and ViT approaches.

---

## 🎥 Live Demo

* **Streamlit App**: [https://autonomouscarapp-app-iur6pfwku8u5pjovsjmcab.streamlit.app/](https://autonomouscarapp-app-iur6pfwku8u5pjovsjmcab.streamlit.app/)
* **Video Walkthrough**: [https://drive.google.com/file/d/1vSDp0IiL0rcNPOlmwP-Dps-3KsKO\_GIY/view?usp=sharing](https://drive.google.com/file/d/1vSDp0IiL0rcNPOlmwP-Dps-3KsKO_GIY/view?usp=sharing)

---

## 👥 Contributors

| Name                | Roll No.  | Role                                 |
| ------------------- | --------- | ------------------------------------ |
| Udit Kandpal        | M24CSE027 | ViT model & AirSimNH data generation |
| Om Patel            | M24CSA019 | Autoencoders & ViT deployment        |
| Vishvaskumar Patel  | M24CSE029 | CNN (AirSimNH) & report writing      |
| Rahul Maurya        | M24CSA025 | CNN (Udacity) & dataset creation     |
| Patil Divya Kailash | M24CSE018 | CNN (Udacity) & documentation        |

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*May, 2025 — Indian Institute of Technology Jodhpur*
