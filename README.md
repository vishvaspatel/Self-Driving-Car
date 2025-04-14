\begin{center}
% {\huge\textbf{\ttitle}}\\
\end{center}
% \section{Dataset}
\begin{center}
\section*{Abstract}
\end{center}

\hspace{1cm}This project targets the improvement of the lateral control system of self-driving cars through the incorporation of state-of-the-art deep learning methods for perception and control. The system uses convolutional neural networks (CNNs) and Vision Transformers (ViTs) to carry out lane detection and object detection based on camera-based data. These perception units are input to the lateral control logic, which steers the vehicle to ensure safe and accurate lane keeping. By integrating visual understanding in real-time with exact control algorithms, the project promises to enhance stability and responsiveness in autonomous driving under various driving environments. Experimental studies prove the suitability of CNN and ViT perception pipelines in efficient lane detection and dynamic object recognition, allowing robust and smooth lateral control in scenarios involving complex road conditions.
    
\clearpage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section*{Objective}

This project aims to develop and evaluate deep learning-based lateral control systems for autonomous vehicles. The primary objectives include:

\begin{itemize}
    \item Implementing and comparing the performance of Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) for lane detection and steering angle prediction using simulated driving datasets (Udacity and AirsimNH).
    \item Designing and applying data preprocessing and augmentation techniques to enhance the robustness and generalization of the models.
    \item Developing a multi-task learning model to simultaneously predict steering and throttle control.
    \item Implementing steering smoothing and throttle reduction techniques to improve the stability and safety of the autonomous vehicle control.
    \item Deploying the trained model in a web application for demonstration and evaluation.
\end{itemize}
\clearpage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\section{Overview of Simulators}

\subsection{Udacity Simulator} This simulator, often integrated with Udacity's online courses, provides focused environments and scenarios designed for learning specific self-driving concepts and algorithms. It typically features simplified physics and sensor models, prioritizing pedagogical clarity and ease of use for educational purposes. The simulator often utilizes Python and libraries specific to the Udacity curriculum.

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.5\textwidth]{udacity.png}
    \caption{Udacity Stimulator}
    \label{fig:simulator}
\end{figure}


\subsection{AirSimNH Simulator} Based on Microsoft AirSim, this simulator offers a more photorealistic and complex environment with detailed physics and support for a wider range of realistic sensors like lidar and radar. It allows for greater flexibility in creating custom scenarios and integrates well with industry-standard tools and frameworks such as ROS and Unreal Engine, making it suitable for more advanced research and development. AirSimNH enables the simulation of diverse and challenging real-world conditions.


\begin{figure}[h!]
    \centering
    \includegraphics[width=0.5\textwidth]{airsimnh.png}
    \caption{AirsimNH Stimulator}
    \label{fig:simulator}
\end{figure}



\section{Dataset: Udacity}

The dataset includes images captured from three different perspectives: center, left, and right cameras of a simulated vehicle, along with corresponding driving data.

\subsection{Image Data}

\begin{itemize}
    \item \textbf{Center Camera:} Positioned directly at the center of the simulated vehicle, capturing the primary view of the road.
    \item \textbf{Left Camera:} Positioned on the left, providing a view that could be helpful for the model to understand potential off-center positions.
    \item \textbf{Right Camera:} Positioned on the right, assisting the model in learning recovery maneuvers if the vehicle is off-center to the right.
\end{itemize}

\subsection{Driving Data}

\begin{itemize}
    \item \textbf{Steering Angle} The primary label for training, representing the angle of the vehicle's steering wheel.
    \item \textbf{Throttle} Indicates the acceleration at each frame.
    \item \textbf{Reverse} A binary feature indicating whether the vehicle is in reverse mode.
    \item \textbf{Speed} Speed data that affects the throttle control.
\end{itemize}

\subsection{Preprocessing Steps}

The following preprocessing steps were applied to the images to prepare them for model training:

\begin{enumerate}
    \item \textbf{Balancing:} To address the skew in steering angles (with a large portion centered around zero), a balancing technique was applied. The steering angles were divided into 25 bins, and the number of samples in bins with excessive representation were reduced to a target of 400 samples per bin.

\begin{figure}[h!]
    \centering
    \begin{minipage}{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{udacity_data_before_balancing.png}
        \caption{Udacity Dataset before Balancing}
        \label{fig:before_balancing}
    \end{minipage}\hfill
    \begin{minipage}{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{udacity_data_after_balancing.png}
        \caption{Udacity Dataset after Balancing}
        \label{fig:after_balancing}
    \end{minipage}
\end{figure}

    
    \item \textbf{Cropping:} Unnecessary portions of the image, such as the sky and car hood, were cropped out to focus on the relevant road section.
    \item \textbf{Color Space Conversion:} The images were transformed to the YUV color space to enhance road features.
    \item \textbf{Gaussian Blur:} A Gaussian blur was applied to reduce noise and smooth the images.
    \item \textbf{Resizing:} The images were resized to $200 \times 66$ pixels to match the input dimensions of the CNN model.
    \item \textbf{Pixel Value Normalization:} Pixel values were normalized by dividing by 255 to scale them to the range $[0, 1]$.
\end{enumerate}

\subsection{Data Augmentation}

Data augmentation techniques were employed to enhance the robustness of the model:

\begin{enumerate}
    \item \textbf{Zooming:} Images were slightly enlarged to simulate the car approaching closer to the scene.
    \item \textbf{Panning:} Images were shifted slightly in the horizontal or vertical direction to simulate lateral or forward shifts.
    \item \textbf{Random Brightness Adjustment:} The brightness of the images was randomly adjusted to mimic varying lighting conditions.
    \item \textbf{Horizontal Flipping:} Images were flipped horizontally to simulate scenarios where the car might be approaching a curve or making turns.
\end{enumerate}

The dataset undergoes several transformations, summarized in Table \ref{tab:dataset_transformations}.

\begin{table}[h!]
    \centering
    \caption{Dataset Transformations}
    \label{tab:dataset_transformations}
    \begin{tabular}{lc}
        \hline
        Transformation & Number of Samples \\
        \hline
        Total initial data & 4053 \\
        Data removed for balancing & 2590 \\
        Remaining data after balancing & 1463 \\
        Training set after data augmentation & 3511 \\
        Validation set after data augmentation & 878 \\
        \hline
    \end{tabular}
\end{table}

The dataset is split into training and validation sets to facilitate model development and evaluation. The initial dataset contains images from the center, left, and right cameras. After balancing and augmenting the data, the dataset is expanded to include multiple views and variations, significantly increasing the total number of images.
\clearpage


\section{Dataset: AirsimNH}

Since there is no publicly available dataset for this simulator, we generated our own dataset consisting of 34,707 images. The AirsimNH dataset comprises the following information:

\subsection{Image Data}

These images were captured from the simulated environment, with only one centered image captured per frame from the simulator.

\subsection{Driving Data}

\begin{itemize}
    \item \textbf{Vehicle Name:} Identifier for the vehicle in the simulation.
    \item \textbf{Throttle} Indicates the acceleration applied to the vehicle.
    \item \textbf{Steering} The steering angle applied to the vehicle, which is the primary label for training.
    \item \textbf{ImageFile} The file path or identifier for the corresponding image.
\end{itemize}

\subsection{Balancing}

The initial dataset exhibited a skew in the distribution of steering angles. To mitigate this, a balancing step was applied by reducing the number of samples for the predominant steering angle (0.0). The distribution of steering angles before and after balancing is summarized in Table \ref{tab:airsim_balancing}.

\begin{table}[h!]
    \centering
    \caption{AirsimNH Steering Angle Distribution Before and After Balancing}
    \label{tab:airsim_balancing}
    \begin{tabular}{lcc}
        \hline
        Steering Angle & Value Counts (Initial) & Value Counts (Balanced) \\
        \hline
        -0.5 & 1124 & 1124 \\
        0.0 & 26012 & 15000 \\
        0.5 & 7571 & 7571 \\
        \hline
    \end{tabular}
\end{table}

\begin{figure}[h!]
    \centering
    \begin{minipage}{0.41\textwidth}
        \centering
        \includegraphics[width=\textwidth]{airsim_dataset_before_balancing.png}
        \caption{AirsimNH Dataset before Balancing}
        \label{fig:before_balancing}
    \end{minipage}\hfill
    \begin{minipage}{0.41\textwidth}
        \centering
        \includegraphics[width=\textwidth]{airsim_dataset_after_balancing.png}
        \caption{AirsimNH Dataset after Balancing}
        \label{fig:after_balancing}
    \end{minipage}
\end{figure}
\clearpage

\section{Approach 1: CNN (Udacity)}

The first approach to autonomous driving implemented on the Udacity simulator in this project uses a Convolutional Neural Network (CNN). CNNs are well-suited for image-based tasks due to their ability to automatically learn hierarchical features from raw pixel data.

\subsection{Model Architecture: NVIDIA CNN with Modifications}

The CNN architecture used in this project is based on the NVIDIA "End to End Learning for Self-Driving Cars" model. The original NVIDIA model was designed to map raw pixels from a single front-facing camera directly to steering commands. It consists of a series of convolutional layers that extract increasingly complex features from the input images, followed by fully connected layers that output the predicted steering angle.


\begin{table}[h!]
    \centering
    \caption{CNN Model Architecture}
    \label{tab:cnn_architecture}
    \begin{tabular}{lll}
        \hline
        Layer (Type) & Output Shape & Param \# \\
        \hline
        conv2d (Conv2D) & (None, 31, 98, 24) & 1,824 \\
        conv2d\_1 (Conv2D) & (None, 14, 47, 36) & 21,636 \\
        conv2d\_2 (Conv2D) & (None, 5, 22, 48) & 43,248 \\
        conv2d\_3 (Conv2D) & (None, 1, 18, 64) & 76,864 \\
        flatten (Flatten) & (None, 1152) & 0 \\
        dense (Dense) & (None, 100) & 115,300 \\
        dense\_1 (Dense) & (None, 50) & 5,050 \\
        dense\_2 (Dense) & (None, 10) & 510 \\
        dense\_3 (Dense) & (None, 1) & 11 \\
        \hline
        \textbf{Total params:} & & \textbf{264,443 (1.01 MB)} \\
        \textbf{Trainable params:} & & \textbf{264,443 (1.01 MB)} \\
        \textbf{Non-trainable params:} & & \textbf{0 (0.00 B)} \\
        \hline
    \end{tabular}
\end{table}


\subsection{Results}

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.5\textwidth]{result_cnn_udacity.png}
    \caption{Result of CNN On udacity Stimulator}
    \label{fig:simulator}
\end{figure}

\clearpage

\section{Approach 2 - Autoencoders (Udacity)}

The second approach to autonomous driving implemented on the Udacity simulator uses autoencoders in conjunction with a CNN.

\subsection{Model Architecture: Autoencoders}

In this approach, an autoencoder is used to process the input images. The encoder part of the autoencoder is then integrated into the CNN architecture. This allows the CNN to work with a more refined feature representation learned by the autoencoder.

\begin{table}[h!]
    \centering
    \caption{Autoencoder Architecture}
    \label{tab:autoencoder_architecture}
    \begin{tabular}{lll}
        \hline
        Layer (Type) & Output Shape & Param \# \\
        \hline
        input\_layer\_3 (InputLayer) & (None, 100, 320, 3) & 0 \\
        conv2d\_11 (Conv2D) & (None, 100, 320, 8) & 224 \\
        max\_pooling2d\_4 (MaxPooling2D) & (None, 50, 160, 8) & 0 \\
        conv2d\_12 (Conv2D) & (None, 50, 160, 8) & 584 \\
        up\_sampling2d\_4 (UpSampling2D) & (None, 100, 320, 8) & 0 \\
        conv2d\_13 (Conv2D) & (None, 100, 320, 3) & 219 \\
        \hline
        \textbf{Total params:} & & \textbf{1,027 (4.01 KB)} \\
        \textbf{Trainable params:} & & \textbf{1,027 (4.01 KB)} \\
        \textbf{Non-trainable params:} & & \textbf{0 (0.00 B)} \\
        \hline
    \end{tabular}
\end{table}

\subsection{Results}

The CNN with autoencoders achieved a lower validation loss compared to the CNN-only model. This indicates that the enhanced feature representation from the autoencoder helped the model make more accurate predictions.

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.5\textwidth]{result_autoencoders_udacity.png}
    \caption{Result of Autoencoders On udacity Stimulator}
    \label{fig:simulator}
\end{figure}

\clearpage

\section{Approach 3: CNN On AirsimNH (Predicting only Sterring)}

The third approach to autonomous driving implemented in this project uses a Convolutional Neural Network (CNN). This model is designed for end-to-end steering angle prediction from RGB camera images in an autonomous driving simulation environment (Airsim). The architecture consists of a feature extraction module followed by a regression module, with weight initialization for improved training stability.

\subsection{CNN Architecture}

\begin{table}[h!]
    \centering
    \caption{Detailed SteeringCNN Architecture}
    \label{tab:steering_cnn_detailed_architecture}
    \begin{tabular}{lll}
        \hline
        Layer (Type) & Output Shape & Param \# \\
        \hline
        Conv2d-1 & [-1, 32, 64, 64] & 2,432 \\
        BatchNorm2d-2 & [-1, 32, 64, 64] & 64 \\
        ReLU-3 & [-1, 32, 64, 64] & 0 \\
        MaxPool2d-4 & [-1, 32, 32, 32] & 0 \\
        Conv2d-5 & [-1, 64, 32, 32] & 18,496 \\
        BatchNorm2d-6 & [-1, 64, 32, 32] & 128 \\
        ReLU-7 & [-1, 64, 32, 32] & 0 \\
        MaxPool2d-8 & [-1, 64, 16, 16] & 0 \\
        Conv2d-9 & [-1, 128, 16, 16] & 73,856 \\
        BatchNorm2d-10 & [-1, 128, 16, 16] & 256 \\
        ReLU-11 & [-1, 128, 16, 16] & 0 \\
        MaxPool2d-12 & [-1, 128, 8, 8] & 0 \\
        Conv2d-13 & [-1, 256, 8, 8] & 295,168 \\
        BatchNorm2d-14 & [-1, 256, 8, 8] & 512 \\
        ReLU-15 & [-1, 256, 8, 8] & 0 \\
        MaxPool2d-16 & [-1, 256, 4, 4] & 0 \\
        AdaptiveAvgPool2d-17 & [-1, 256, 4, 4] & 0 \\
        Linear-18 & [-1, 512] & 2,097,664 \\
        BatchNorm1d-19 & [-1, 512] & 1,024 \\
        ReLU-20 & [-1, 512] & 0 \\
        Dropout-21 & [-1, 512] & 0 \\
        Linear-22 & [-1, 256] & 131,328 \\
        BatchNorm1d-23 & [-1, 256] & 512 \\
        ReLU-24 & [-1, 256] & 0 \\
        Dropout-25 & [-1, 256] & 0 \\
        Linear-26 & [-1, 1] & 257 \\
        \hline
        \textbf{Total params:} & & \textbf{2,621,697} \\
        \textbf{Trainable params:} & & \textbf{2,621,697} \\
        \textbf{Non-trainable params:} & & \textbf{0} \\
        \hline
    \end{tabular}
\end{table}

\subsection{Steering-Dependent Throttle Reduction:}
The throttle is reduced proportionally to the magnitude of the steering angle to slow the vehicle during turns. The reduction is governed by a parameter, $R = 0.2$, termed the steering throttle reduction factor. The adjusted throttle, $T_{\text{adjusted}}$, is calculated using the formula:
\[
T_{\text{adjusted}} = T_{\text{base}} \cdot (1 - |\theta| \cdot R)
\]
where:
\begin{itemize}
    \item $\theta$ is the smoothed and clipped steering angle, $\theta \in [-1.0, 1.0]$,
    \item $|\theta|$ is the absolute value of the steering angle,
    \item $R = 0.2$ is the reduction factor.
\end{itemize}

\subsection{Steering Smoothing:}
The raw steering prediction from the CNN is smoothed using an exponential moving average to reduce jitter and ensure smooth control transitions. The smoothing is implemented as:
\[
\theta_{\text{smoothed}} = \alpha \cdot \theta_{\text{new}} + (1 - \alpha) \cdot \theta_{\text{prev}}
\]
where:
\begin{itemize}
    \item $\alpha = 0.3$ is the smoothing factor,
    \item $\theta_{\text{new}}$ is the current predicted steering angle,
    \item $\theta_{\text{prev}}$ is the previously smoothed steering angle.
\end{itemize}
The smoothed steering angle, $\theta_{\text{smoothed}}$, is clipped to the range [-1.0, 1.0] to ensure it remains within the vehicle’s steering limits.


\subsection{Result}

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.6\textwidth]{cnn_single_Loss_curv.png}
    \caption{Result of CNN On AirsimNH Stimulator prediction steering only}
    \label{fig:simulator}
\end{figure}

\clearpage

\section{Approach: 4 CNN On AirsimNH (Predicting Steering and Throttle)}

The fourth approach implemented in this project uses a Convolutional Neural Network (CNN) to predict both steering and throttle values. This model is designed for end-to-end control of an autonomous vehicle in the Airsim simulation environment.

\subsection{CNN Architecture}

The CNN architecture for predicting both steering and throttle is detailed in Table \ref{tab:steering_throttle_cnn_architecture}.

\begin{table}[h!]
    \centering
    \caption{CNN Architecture for Steering and Throttle Prediction}
    \label{tab:steering_throttle_cnn_architecture}
    \begin{tabular}{lll}
        \hline
        Layer (Type) & Output Shape & Param \# \\
        \hline
        Conv2d-1 & [-1, 32, 64, 64] & 2,432 \\
        BatchNorm2d-2 & [-1, 32, 64, 64] & 64 \\
        ReLU-3 & [-1, 32, 64, 64] & 0 \\
        MaxPool2d-4 & [-1, 32, 32, 32] & 0 \\
        Conv2d-5 & [-1, 64, 32, 32] & 18,496 \\
        BatchNorm2d-6 & [-1, 64, 32, 32] & 128 \\
        ReLU-7 & [-1, 64, 32, 32] & 0 \\
        MaxPool2d-8 & [-1, 64, 16, 16] & 0 \\
        Conv2d-9 & [-1, 128, 16, 16] & 73,856 \\
        BatchNorm2d-10 & [-1, 128, 16, 16] & 256 \\
        ReLU-11 & [-1, 128, 16, 16] & 0 \\
        MaxPool2d-12 & [-1, 128, 8, 8] & 0 \\
        Conv2d-13 & [-1, 256, 8, 8] & 295,168 \\
        BatchNorm2d-14 & [-1, 256, 8, 8] & 512 \\
        ReLU-15 & [-1, 256, 8, 8] & 0 \\
        MaxPool2d-16 & [-1, 256, 4, 4] & 0 \\
        AdaptiveAvgPool2d-17 & [-1, 256, 4, 4] & 0 \\
        Linear-18 & [-1, 512] & 2,097,664 \\
        BatchNorm1d-19 & [-1, 512] & 1,024 \\
        ReLU-20 & [-1, 512] & 0 \\
        Dropout-21 & [-1, 512] & 0 \\
        Linear-22 & [-1, 256] & 131,328 \\
        BatchNorm1d-23 & [-1, 256] & 512 \\
        ReLU-24 & [-1, 256] & 0 \\
        Dropout-25 & [-1, 256] & 0 \\
        Linear-26 (Steering) & [-1, 1] & 257 \\
        Linear-27 (Throttle) & [-1, 256] & 131,328 \\
        BatchNorm1d-28 & [-1, 256] & 512 \\
        ReLU-29 & [-1, 256] & 0 \\
        Dropout-30 & [-1, 256] & 0 \\
        Linear-31 (Throttle) & [-1, 1] & 257 \\
        \hline
        \textbf{Total params:} & & \textbf{2,953,794} \\
        \textbf{Trainable params:} & & \textbf{2,953,794} \\
        \textbf{Non-trainable params:} & & \textbf{0} \\
        \hline
    \end{tabular}
\end{table}

\subsection{Multi-Task Loss Function for Steering and Throttle Prediction}

In the proposed model, a custom multi-task loss function is implemented to jointly optimize predictions for steering and throttle control. The loss function uses mean squared error (MSE) to quantify prediction errors for both steering and throttle outputs. The total loss is a weighted sum of the individual steering and throttle losses:
\[
\text{Total Loss} = w_s \cdot \text{Steering Loss} + w_t \cdot \text{Throttle Loss}
\]
where $w_s$ and $w_t$ are hyperparameters that modulate the relative importance of each task. This multi-task loss formulation ensures that the model optimizes both steering and throttle predictions simultaneously.

\subsection{Steering Smoothing:}

The same steering smoothing technique was used as in the previous approach.

\subsection{Result}

\begin{figure}[h!]
    \centering
    \includegraphics[width=1\textwidth]{cnn_double_loss_curves.png}
    \caption{Result of CNN On AirsimNH Stimulator prediction steering and Throttle}
    \label{fig:simulator}
\end{figure}

\clearpage



\section{Approach 5: ViT On AirsimNH (Predicting Steering Only)}

This research investigates an end-to-end learning approach for predicting steering angles in the AirsimNH simulation environment using a Vision Transformer (ViT) architecture.

\subsection{Detailed Layer Summary:}

The following table provides a layer-wise breakdown of the network architecture, including approximate parameter counts based on the standard \texttt{vit\_small\_patch16\_224} configuration:

\begin{table}[h!]
    \centering
    \caption{ViT Architecture for Steering Prediction}
    \label{tab:vit_architecture}
    \begin{tabular}{llll}
        \hline
        Layer & Input Shape & Output Shape & Parameters \\
        \hline
        Backbone (ViT) & $\sim$21.6M \\
        \hspace{1em} Patch Embedding & (3, 224, 224) & (196, 384) & $\sim$590K \\
        \hspace{1em} Transformer Blocks (12) & (196, 384) & (196, 384) & $\sim$21M \\
        \hspace{1em} Pooling/Norm & (196, 384) & (384,) & $\sim$0.8K \\
        Head & & & $\sim$132K \\
        \hspace{1em} LayerNorm & (384,) & (384,) & 768 \\
        \hspace{1em} Linear & (384,) & (256,) & 98,560 \\
        \hspace{1em} GELU & (256,) & (256,) & 0 \\
        \hspace{1em} Dropout (0.3) & (256,) & (256,) & 0 \\
        \hspace{1em} Linear & (256,) & (128,) & 32,896 \\
        \hspace{1em} GELU & (128,) & (128,) & 0 \\
        \hspace{1em} Dropout (0.2) & (128,) & (128,) & 0 \\
        \hspace{1em} Linear & (128,) & (1,) & 129 \\
        \hline
        \textbf{Total Parameters} & & & $\sim$21.73M \\
        \hline
    \end{tabular}
\end{table}

\subsection{Steering Smoothing:}

The same steering smoothing technique was used as in the previous approach.

\subsection{Steering-Dependent Throttle Reduction:}

The same technique was used as in the previous approach.

\subsection{Result}

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.5\textwidth]{vit_single_loss_curve.png}
    \caption{Result of ViT On AirsimNH Stimulator prediction steering}
    \label{fig:simulator}
\end{figure}

\clearpage


\section{Approach 6: ViT on AirsimNH (Predicting Steering and Throttle)}

This approach utilizes a Vision Transformer (ViT) architecture to simultaneously predict both steering and throttle values for autonomous navigation in the AirsimNH simulation environment. The model takes raw RGB images as input and outputs continuous values for both steering angle and throttle.

\begin{table}[h!]
    \centering
    \caption{ViT Architecture for Steering and Throttle Prediction}
    \label{tab:vit_steering_throttle_architecture}
    \begin{tabular}{llll}
        \hline
        \textbf{Layer Name} & \textbf{Type} & \textbf{Input Shape} & \textbf{Output Shape} \\
        \hline
        backbone & Vision Transformer (ViT) & (3, 224, 224) & (384,) \\
        shared\_head[0] & LayerNorm & (384,) & (384,) \\
        shared\_head[1] & Linear & (384,) & (256,) \\
        shared\_head[2] & GELU & (256,) & (256,) \\
        shared\_head[3] & Dropout (0.3) & (256,) & (256,) \\
        steering\_head[0] & Linear & (256,) & (128,) \\
        steering\_head[1] & GELU & (128,) & (128,) \\
        steering\_head[2] & Dropout (0.2) & (128,) & (128,) \\
        steering\_head[3] & Linear & (128,) & (1,) \\
        throttle\_head[0] & Linear & (256,) & (128,) \\
        throttle\_head[1] & GELU & (128,) & (128,) \\
        throttle\_head[2] & Dropout (0.2) & (128,) & (128,) \\
        throttle\_head[3] & Linear & (128,) & (1,) \\
        sigmoid & Sigmoid Activation & (1,) & (1,) \\
        \hline
    \end{tabular}
\end{table}


\subsection{Steering Smoothing:}
The same steering smoothing technique was used as in the previous approach.

\subsection{Result}
\begin{figure}[h!]
    \centering
    \includegraphics[width=1\textwidth]{vit_double_loss_curves.png}
    \caption{Result of ViT On AirsimNH Stimulator prediction steering and Throttle}
    \label{fig:simulator}
\end{figure}

\clearpage
\section{Approach 7: Lateral Control with Object Detection and Lane Detection on AirSimNH}

\subsection{Lateral Control Model}

The lateral control model used in this approach is the same as described in Approach 5. We employed a Vision Transformer (ViT)-based architecture for steering angle prediction, enhanced with techniques such as steering smoothing and Steering-Dependent Throttle Reduction to improve stability and safety during autonomous driving.

\subsection{Object Detection using YOLOv5}

To enhance safety and support obstacle avoidance, we integrated the YOLOv5s object detection model. Pretrained on the COCO dataset, YOLOv5 detects various objects including cars, buses, and trucks. Only high-confidence detections of relevant classes are considered. When an object is detected within a predefined collision zone in the image, the system dynamically adjusts the steering angle and activates emergency braking to prevent potential collisions.

\subsection{Lane Detection Module}

A lightweight lane detection system was implemented using traditional computer vision techniques such as Canny edge detection and Hough Line Transform. These detected lane lines are overlaid for visualization and provide contextual support to the transformer model during real-time navigation.

\subsection{Demo of Lateral Control with Object Detection and Lane Detection on AirSimNH}

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.8\textwidth]{Screenshot 2025-04-14 142626.png}
    \caption{Demo of Lateral Control with Object Detection and Lane Detection on AirSimNH}
    \label{fig:simulator}
\end{figure}

\clearpage


\section*{Deployment}

The trained model has been deployed as a web application to allow for interactive demonstration and testing. The application provides a user interface to visualize the model's predictions and control commands. The deployment can be accessed at the following link:
\\
\url{https://requirementstxt-vhoxy8j2hvuzsqcfbesfxj.streamlit.app/}

\section*{Code Repository}

The complete source code for this project, including model implementations, training scripts, and data processing utilities, is available on GitHub. The repository contains detailed documentation and instructions for reproducing the experiments and utilizing the codebase. Access the code at:
\\
\url{https://github.com/vishvaspatel/Lateral-Control-for-Autonomous-Vehicles-Self-Driving-Car-}


\section*{Deployed Website Demo}

\url{https://requirementstxt-vhoxy8j2hvuzsqcfbesfxj.streamlit.app/}

\section*{Deployment Website ScreenShot}
\begin{figure}[h!]
    \centering
    \includegraphics[width=1\textwidth]{Screenshot 2025-04-14 125503.png}
    \caption{Website Demo}
    \label{fig:simulator}
\end{figure}

\section*{Video Demo}

\url{https://drive.google.com/file/d/1KY-AiZtJS-uPN8IbmP-pbl9t1usOdNmM/view?usp=drive_link}


\section*{Contribution}

\begin{itemize}
    \item \textbf{Udit Kandpal (M24CSE027):} Udit contributed to the project by working on the Vision Transformer (ViT) model, including training the model and generating data within the AirsimNH simulator.
    \item \textbf{Om Patel (M24CSA019):} Om's contribution involved autoencoder model development for Udacity and training and deployment of the Vision Transformer (ViT) model within the AirsimNH environment.
    \item \textbf{Vishvaskumar Patel (M24CSE029):} Vishvaskumar was responsible for the Convolutional Neural Network (CNN) model training within the AirsimNH simulator environment and also played a key role in report writing.
    \item \textbf{Rahul Maurya (M24CSA025):} Rahul's work involved data generation and the training of the CNN conducted using the Udacity simulator.
    \item \textbf{Patil Divya Kailash (M24CSE018):} Divya focused on the Convolutional Neural Network (CNN) model training, specifically utilizing the Udacity simulator and Report writing for this project.
\end{itemize}
