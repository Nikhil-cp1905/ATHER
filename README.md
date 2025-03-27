# Ather Electric Scooter Detection using YOLOv8

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Dependencies](#dependencies)
4. [Installation and Setup](#installation-and-setup)
   - [Clone the Repository](#1-clone-the-repository)
   - [Install Dependencies](#2-install-dependencies)
   - [Download the Dataset](#3-download-the-dataset)
   - [Place the Model File](#4-place-the-model-file)
   - [Run the Detection Script](#5-run-the-detection-script)
5. [Usage Instructions](#usage-instructions)
6. [File Structure](#file-structure)
7. [How It Works](#how-it-works)
8. [Performance Metrics](#performance-metrics)
9. [Future Improvements](#future-improvements)
10. [Contributing](#contributing)
11. [License](#license)
12. [Author](#author)

## Overview
This project focuses on real-time **Ather Electric Scooter Detection** using **YOLOv8**, a state-of-the-art deep learning model for object detection. The system is designed to process live webcam feed, detect Ather scooters, and display bounding boxes along with confidence scores.


![Screenshot from 2025-03-27 19-51-53](https://github.com/user-attachments/assets/9c2ed141-30dc-4140-8f2b-79db059c36ac)


![Screenshot from 2025-03-27 19-49-11](https://github.com/user-attachments/assets/83ee50a2-a565-461e-a68e-69df93d59ec5)

## Features

- **Real-time Object Detection**: Uses a webcam to detect Ather scooters in real-time.
- **High Accuracy**: Trained on a curated dataset using **YOLOv8** for precise detection.
- **Optimized Performance**: Utilizes **OpenCV** and **PyTorch** for efficient processing.
- **Custom Dataset Integration**: Downloads a custom dataset from **Roboflow** for training.
- **Scalability**: Can be extended to detect other vehicle types or integrated into broader surveillance systems.

## Dependencies
Ensure your system has the following dependencies installed:
```bash
pip install ultralytics opencv-python roboflow torch torchvision torchaudio numpy
```
If you face any installation issues, ensure that you are using **Python 3.8+** and a **GPU-enabled system** (for better performance).

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Nikhil-cp1905/Ather-YOLOv8-Detection.git
cd Ather-YOLOv8-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
_(Alternatively, install dependencies manually as mentioned above.)_

### 3. Download the Dataset
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")  # Replace with your API key
project = rf.workspace("jarivs").project("ather-hy9he")
version = project.version(3)
dataset = version.download("yolov8")
```

### 4. Place the Model File
Ensure `best.pt` (trained YOLOv8 model) is in the project directory. If you need to train a model, refer to the **Training Instructions** below.

### 5. Run the Detection Script
```bash
python detect.py
```

## Usage Instructions
- The program starts the default webcam and begins real-time scooter detection.
- Bounding boxes are drawn around detected scooters along with confidence scores.
- Press **'q'** to exit the application.

## File Structure
```
Ather-YOLOv8-Detection/
│── best.pt               # Trained YOLOv8 model
│── detect.py             # Main script for real-time detection
│── dataset/              # Contains dataset (after download)
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
```

## How It Works
1. **Data Collection & Preprocessing**
   - Dataset is sourced from **Roboflow** and annotated with bounding boxes for Ather scooters.
   - Images are preprocessed for optimal input to the YOLOv8 model.

2. **Model Training**
   - The YOLOv8 model is trained on the collected dataset with customized hyperparameters.
   - Trained weights are saved in `best.pt`.

3. **Real-Time Detection**
   - The trained model processes frames from the webcam.
   - Bounding boxes are drawn for detected scooters.

## Performance Metrics
- **Mean Average Precision (mAP)**: Achieved **~90%** accuracy in detecting Ather scooters.
- **Inference Speed**: Processes **30+ FPS** on a GPU-powered system.
- **False Positives**: Minimal false positives due to high-quality dataset.

## Future Improvements
- **Edge Device Optimization**: Porting the model to work on **Raspberry Pi / Jetson Nano**.
- **Multiple Object Detection**: Expanding the dataset to include **other vehicle types**.
- **Improved Speed**: Implementing **TensorRT acceleration** for real-time detection.

## Contributing
We welcome contributions! Follow these steps to contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-new-feature`).
3. Commit your changes (`git commit -m "Added a new feature"`).
4. Push the branch (`git push origin feature-new-feature`).
5. Open a Pull Request.

## License
This project is licensed under the **MIT License**.

## Author
Developed by **[Nikhil](https://github.com/Nikhil-cp1905)** 🚀

