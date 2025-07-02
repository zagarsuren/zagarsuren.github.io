## HerdWatch: Real-time Object Detection for Livestock management

### Project Overview:
HerdWatch is a deep learning-powered livestock monitoring system designed to automate cattle behaviour detection and counting using computer vision. Addressing inefficiencies in traditional livestock management, the project leverages object detection models to classify behaviours such as estrus, grazing, lying, and standing from images and video. The system is tailored for integration with surveillance cameras and drones, enabling real-time analysis for health, reproductive, nutritional, and productivity insights.

### Key Objectives:
- Automate behaviour recognition (e.g., estrus, grazing) to support early disease detection and reproductive health tracking.
- Enhance farm productivity and animal welfare through AI-powered monitoring.
Evaluate and compare multiple object detection architectures to identify the most effective model for livestock applications.
- Build a user-friendly web-based interface supporting image/video upload and real-time webcam detection.

### Core Technologies:
- Object Detection: YOLOv5, YOLOv8 (nano & medium), Faster R-CNN
- Dataset: CowBehaviorDetection (Roboflow)
- Annotation Tools: CVAT.ai
- Web Development: Django, Wagtail, HTML/CSS, JavaScript
- Deployment: Local-hosted demo via image and live feed upload

### Summary of Results:
- YOLOv8 Medium achieved the best performance with mAP@0.5 = 0.906 and overall AP of 0.894, outperforming all other models.
- Faster R-CNN performed well on large cattle images but struggled with real-time inference speed.
- The system accurately classified 4 key behaviours, with future potential to scale to additional categories.
- A responsive GUI was implemented to support real-time webcam detection, image/video upload, and interactive result display.
- Identified key limitations, including poor small object detection, occlusion sensitivity, and hardware demands for training/inference.

### Skills: 
> Object Detection · Object-Oriented Programming (OOP) · Python (Programming Language) · PyTorch · Django · UI/UX · YOLOv8 · Faster R-CNN 

### System Architecture
![img](https://i.imgur.com/1iuKYYp.png)

### Graphical User Interface
![img](https://i.imgur.com/DweXM3J.png)
![img](https://i.imgur.com/87ku0My.png)