# 🛩️ Aerial Object Classification & Detection

### Deep Learning-Based Bird and Drone Classification & Detection

Aerial Object Classification & Detection is a deep learning project designed to distinguish birds from drones and detect their locations within images. The system combines image classification, object detection, and visual explainability in a Streamlit application called **Aerial Surveillance AI**.

The project evaluates multiple deep learning architectures and uses ResNet50V2 for classification and YOLOv8 for object detection.

## 📌 Features

### 🐦 Bird and Drone Classification

Classifies aerial images into two categories:

- Bird
- Drone

The final classification model uses ResNet50V2 and achieved 98.2% accuracy in the reported evaluation.

### 🎯 Object Detection

Uses YOLOv8 to identify and localize objects within an image using bounding boxes.

The reported detection performance is an mAP@0.5 score of 0.82.

### 🧠 Deep Learning Model Comparison

Evaluates four architectures to identify suitable models for classification:

- Custom CNN
- EfficientNetB0
- ResNet50V2
- MobileNetV2

### 🔍 Explainable AI

Uses Grad-CAM heatmaps to visualize the image regions influencing classification predictions.

The report describes heatmaps highlighting bird wings and drone bodies or propellers.

### 🖥️ Interactive Web Application

Provides a Streamlit application with a dark-mode command-center interface for classification and object detection.

### 📈 Data Augmentation

Uses image augmentation techniques to improve model generalization across different object orientations and perspectives.

## 🏗️ System Architecture

```text
             Input Aerial Image
                     │
                     ▼
             Image Preprocessing
                     │
                     ▼
              ┌──────┴──────┐
              │             │
              ▼             ▼
        ResNet50V2         YOLOv8
        Classifier         Detector
              │             │
              ▼             ▼
        Bird / Drone    Bounding Boxes
        Classification  and Locations
              │             │
              └──────┬──────┘
                     │
                     ▼
              Streamlit App
                     │
                     ▼
            Results and Visuals
```

## 🔄 Application Workflow

```text
Aerial Image Dataset
         ↓
Exploratory Data Analysis
         ↓
Image Preprocessing
         ↓
Data Augmentation
         ↓
Train and Evaluate Models
         ↓
Select Best Classifier
         ↓
ResNet50V2 Classification
         ↓
YOLOv8 Object Detection
         ↓
Grad-CAM Explainability
         ↓
Streamlit Deployment
```

## 🤖 Deep Learning Models

The project addresses two related computer vision tasks: image classification and object detection.

### 1. Bird and Drone Classification

**Problem Type:** Binary Image Classification

**Selected Model:** ResNet50V2

The classifier distinguishes between birds and drones by learning visual features from aerial images.

The project compares four architectures:

- **Custom CNN:** A four-block convolutional neural network used as the baseline.
- **EfficientNetB0:** Evaluated for its compound-scaling approach.
- **ResNet50V2:** Uses residual connections for deep feature extraction.
- **MobileNetV2:** A lightweight architecture evaluated for potential edge-device deployment.

### 2. Object Detection

**Model:** YOLOv8

While the classifier predicts the class of an image, YOLOv8 identifies object locations and draws bounding boxes around detected objects.

The detection component supports visual localization within an image.

## 📊 Model Performance

The following results are reported in the project PDF.

### Classification Performance

| Model | Accuracy | F1-Score | AUC |
|---|---:|---:|---:|
| Custom CNN | 87.0% | 0.86 | 0.948 |
| EfficientNetB0 | 73.0% | 0.71 | 0.768 |
| ResNet50V2 | 98.2% | 0.98 | 0.998 |
| MobileNetV2 | 98.0% | 0.98 | 0.998 |

**ResNet50V2** achieved the highest reported classification accuracy, while **MobileNetV2** provided comparable performance with a lightweight architecture.

### Object Detection Performance

| Metric | Result |
|---|---:|
| Model | YOLOv8 |
| mAP@0.5 | 0.82 |

The report describes the YOLOv8 detector as the component responsible for locating objects with bounding boxes.

### ResNet50V2 Evaluation

According to the report, the confusion matrix showed four classification errors across 215 test images.

## 📊 Dataset and Exploratory Data Analysis

The project report describes a dataset containing 3,319 images.

Key findings include:

- All training images were reported to have dimensions of 416 × 416 pixels.
- The dataset contained 1,414 bird images and 1,248 drone images in the reported class distribution.
- The dataset had a slight class imbalance.
- Birds and drones presented a visual similarity challenge because both could appear as dark silhouettes against the sky.

The report describes using stratified sampling and data augmentation to address the class distribution and improve generalization.

## 🧹 Image Preprocessing and Augmentation

### Image Preprocessing

Pixel values are rescaled from the range `[0, 255]` to `[0, 1]`.

The report also describes resizing images to 224 × 224 pixels for the classification model.

### Data Augmentation

The image augmentation pipeline uses:

- **Rotation:** Up to 20 degrees.
- **Zoom and shear:** Approximately 10%.
- **Horizontal flipping:** To account for changes in object orientation.

These transformations help the model learn visual patterns across different orientations and perspectives.

## 🔍 Explainable AI with Grad-CAM

The project uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize the image regions that influence the ResNet50V2 classifier.

The report describes heatmaps that highlight:

- Bird wings and structural features.
- Drone bodies and propellers.

This provides a visual explanation of the classifier's predictions and helps assess whether the model is focusing on relevant object features.

## 🖥️ Application

The project is presented through a Streamlit web application named **Aerial Surveillance AI**.

According to the report, the application:

- Provides a dark-mode command-center interface.
- Integrates the ResNet classifier.
- Integrates the YOLOv8 object detector.
- Presents classification results and object localization.
- Demonstrates Grad-CAM-based visual explanations.

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow / Keras |
| Image Classification | ResNet50V2 |
| Additional Architectures | Custom CNN, EfficientNetB0, MobileNetV2 |
| Object Detection | YOLOv8 |
| Image Processing | Image preprocessing and augmentation |
| Explainable AI | Grad-CAM |
| Web Application | Streamlit |
| Version Control | Git / GitHub |

## 🚀 Setup and Deployment

The project report documents the Streamlit deployment and the trained model architectures but does not provide a complete installation guide, dependency list, or verified local run commands.

For local setup:

1. Clone the repository.
2. Review the repository's dependency files and install the required packages.
3. Ensure the trained classification and detection model artifacts are available.
4. Follow the application's entry point and configuration in the repository to launch the Streamlit interface.

### Clone the Repository

```bash
git clone https://github.com/PiyushChavda595/Aerial-Object-Classification-Detection.git
```

```bash
cd Aerial-Object-Classification-Detection
```

### Live Application

The report identifies the deployed application as **Aerial Surveillance AI · Streamlit**, but does not provide its direct URL.

## ⚠️ Limitations and Considerations

- Classification and detection results depend on the images presented to the models.
- Birds and drones can have similar visual silhouettes, making classification challenging.
- The reported classification and detection metrics describe the project's evaluation and do not guarantee the same performance in every real-world environment.
- The report discusses edge deployment as a future possibility rather than documenting a completed edge-device deployment.
- The project report does not provide a verified real-time CCTV integration.

## 🔮 Future Scope

The project report identifies several possible improvements.

### 📹 Video Stream Integration

Extend the application to process RTSP video feeds from CCTV cameras for continuous monitoring.

### ⚡ Edge Deployment

Convert MobileNetV2 to TensorFlow Lite (TFLite) to support inference directly on compatible drone hardware without requiring an internet connection.

### 🔊 Audio Integration

Combine visual detection with acoustic sensors to analyze drone propeller sounds and provide additional information for aerial object identification.

## 👨‍💻 Project Information

- **Project:** Aerial Object Classification & Detection
- **Application Name:** Aerial Surveillance AI
- **Author:** Piyush Chavda
- **Report Date:** November 28, 2025
- **Purpose:** Bird and drone classification, object detection, and visual explainability
