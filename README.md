
# Fall Detection System using YOLO

This project implements a real-time fall detection system using the YOLO (You Only Look Once) deep learning framework. Our goal is to accurately detect fall events from video streams in real-time, making the system applicable in environments such as elderly care, hospitals, or public surveillance.

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Project Architecture](#project-architecture)
- [Datasets](#datasets)
- [Model Details](#model-details)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Future Work](#future-work)
- [References](#references)
- [Acknowledgements](#acknowledgements)

---

## Overview

This fall detection system leverages the power of YOLO—one of the most popular real-time object detection models—to identify falls in video frames. By fine-tuning YOLOv11 on the LE2I dataset (among other sources), the model is designed to distinguish between fall and non-fall activities with high recall and precision. The system processes video input, detects human figures, and determines if a fall event has occurred, alerting caregivers or logging the event for further analysis.

---

## Motivation

Falls are one of the leading causes of injury, especially among the elderly. Early detection and rapid response can significantly reduce the risk of severe injury. This project aims to:
- Improve real-time fall detection accuracy using state-of-the-art object detection techniques.
- Provide a scalable solution that can be deployed in healthcare settings, public spaces, and assisted living environments.
- Serve as a foundation for future research into integrating additional modalities (e.g., sensor data, behavioral analysis) into fall detection systems.

---

## Project Architecture

The system is composed of the following components:
- **Video Input Module**: Captures real-time video feed from cameras.
- **Preprocessing Pipeline**: Resizes and normalizes video frames.
- **YOLO-based Object Detection**: Uses YOLOv11 to detect human figures and localize them with bounding boxes.
- **Fall Classification**: Applies additional logic or post-processing to distinguish between fall events and other activities.
- **Alert/Logging Module**: Records the detected events or sends alerts to caregivers.

### Workflow Diagram

```
+--------------------+       +-------------------+      +----------------------+
| Video Input Module | ----> | Preprocessing &   | ---> | YOLOv11 Object       |
| (Cameras/Feed)     |       | Data Augmentation |      | Detection            |
+--------------------+       +-------------------+      +----------------------+
                                                       |
                                                       v
                                         +----------------------------+
                                         | Fall Classification Logic  |
                                         | (Bounding Box Analysis,    |
                                         | Confidence Scoring, etc.)  |
                                         +----------------------------+
                                                       |
                                                       v
                                        +-----------------------------+
                                        | Attendance/Alert Logging    |
                                        | (Database, SMS/Email Alerts)|
                                        +-----------------------------+
```

---

## Datasets

### LE2I Dataset
- **Description**: The LE2I dataset is specifically designed for fall detection. It contains videos of real-life fall scenarios recorded from multiple angles, providing annotated data for both fall and non-fall activities.
- **Usage**: We fine-tuned YOLOv11 using this dataset to capture the variability of fall events in different conditions.
- **Link**: [LE2I Dataset](https://universe.roboflow.com/le2iahlam/le2i-ahlam/model/1)

### Additional Datasets (Optional)
- **Fall Detection Dataset (IMVIA)**: Contains sensor-based data that can be useful for multi-modal approaches.
- **UCF Fall Detection Dataset**: A collection of videos simulating falls in various settings.

---

## Model Details

The YOLOv11 model used in this project is fine-tuned to detect falls with high recall and precision. Key aspects include:

- **Box Loss**: Optimizes the bounding box regression by comparing predicted boxes with ground truth using metrics such as CIoU or GIoU.
- **Class Loss**: Ensures the correct classification of detected objects (fall vs. non-fall) using techniques like Binary Cross-Entropy or Focal Loss.
- **Distribution Focal Loss (DFL)**: Refines the localization by predicting a probability distribution over possible bounding box coordinates.
- **Fine-Tuning**: We adjust the model weights using the LE2I dataset to adapt YOLOv11’s general object detection capabilities to the specialized task of fall detection.

---

## Installation

### Prerequisites
- Python 3.7 or later
- PyTorch (version compatible with your YOLO implementation)
- OpenCV
- Other dependencies as listed in `requirements.txt`

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/fall-detection-yolo.git
   cd fall-detection-yolo
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Weights:**
   - Download YOLOv11 weights from [Ultralytics](https://docs.ultralytics.com/models/yolo11/) or your specified source.
   - Place the weights file in the designated folder (e.g., `./weights/`).

---

## Usage

### Running the Detection System

1. **Real-time Detection:**
   ```bash
   python detect.py --source 0 --weights ./weights/yolo11n.pt --conf 0.4
   ```
   - `--source 0` uses the webcam.
   - Adjust the `--conf` threshold as needed.

2. **Batch Processing of Video Files:**
   ```bash
   python detect.py --source path/to/video.mp4 --weights ./weights/yolo11n.pt --conf 0.4
   ```

3. **Logging Attendance/Alerts:**
   - The system logs each detection event along with a timestamp.
   - Modify the logging module to connect to your database or messaging service.

---

## Training

To fine-tune the model on your dataset:

1. **Data Preparation:**
   - Annotate your dataset using YOLO annotation format.
   - Organize your dataset folder structure as required by the training script.

2. **Run Training:**
   ```bash
   python train.py --data data.yaml --weights ./weights/yolo11.pt --epochs 50 --batch-size 16
   ```
   - Ensure your `data.yaml` file correctly points to your training and validation data.

3. **Monitor Training:**
   - Use TensorBoard or similar tools to track loss metrics and model performance.

---

## Evaluation

After training, evaluate the model using:
```bash
python val.py --data data.yaml --weights runs/train/exp/weights/best.pt
```
Key metrics include:
- Accuracy
- Sensitivity
- Specificity
- Precision & Recall

These metrics help you understand how well the model distinguishes fall events from normal activities.

---

## Future Work

- **Multi-modal Integration**: Combine video data with sensor inputs for robust fall detection.
- **Edge Deployment**: Optimize and deploy the model on edge devices for real-time applications in hospitals or care facilities.
- **User Interface**: Develop a web or mobile application to monitor real-time fall detection and manage attendance logs.
- **Additional Features**: Integrate mask detection, action recognition, and behavior analysis for enhanced safety monitoring.

---

## References

1. **LE2I Dataset**: [LE2I Fall Detection Dataset](https://universe.roboflow.com/le2iahlam/le2i-ahlam/model/1)
2. **YOLOv11 Documentation**: [Ultralytics YOLOv11](https://docs.ultralytics.com/models/yolo11/)
3. **YOLOv8 Information**: [YOLOv8](https://yolov8.com/)
4. Additional literature on loss functions (Box Loss, Class Loss, Distribution Focal Loss) and fine-tuning methods for object detection.

---

## Acknowledgements

We acknowledge the contributions of the research community in advancing object detection technologies and thank the maintainers of the LE2I dataset, YOLO frameworks, and associated libraries. Special thanks to our advisors and collaborators who provided feedback during the development of this project.

---

Feel free to modify any sections as per your project’s specifics and add any additional details (such as screenshots, diagrams, or further instructions) that can help users understand and utilize your system effectively.

---
