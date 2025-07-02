## AI-Powered Disaster Assessment and Cost Estimation System

### Project Overview:
DeployForce is an AI-powered disaster response solution that automates building damage assessment from satellite imagery using deep learning. It leverages semantic segmentation and classification to turn pre- and post-disaster high-resolution images into colour-coded damage maps and downloadable PDF reports, enabling rapid response and cost estimation. The system is trained on the xBD Dataset, a large-scale annotated satellite imagery dataset designed for building damage detection across multiple global disasters.

### Key Objectives:
- Automate segmentation and classification of damaged buildings using high-resolution satellite images.
- Develop a real-time web app for batch analysis and geolocated severity mapping.
- Enable emergency teams to generate instant cost-estimation reports for recovery planning.
- Implement full CI/CD and MLOps pipelines to support continuous training and deployment.

### Core Technologies:
- Semantic Segmentation with YOLOv9
- xBD Dataset (Annotated satellite imagery for building damage assessment)
- Python, PyTorch, ClearML, Optuna, Docker, AWS S3
- Microservices architecture using Django + Flask
- CI/CD integration with GitHub Actions
- Geospatial visualisation and PDF reporting

### Summary of Results:
Model Performance (on validation set):

Segmentation Only:
- Precision: 0.799
- Recall: 0.895
- F1 Score: 0.842
  
Segmentation + Classification:
- Precision: 0.641
- Recall: 0.748
- F1 Score: 0.678

### System Overview
![img](https://i.imgur.com/jp1fabi.png)


![img](https://i.imgur.com/IfohMTo.jpeg)

### User Interface Design
![img](https://i.imgur.com/aPcvgFC.jpeg)

![img](https://i.imgur.com/1h8Zbhz.jpeg)

### ClearML Pipeline
![img](https://i.imgur.com/YGsa554.jpeg)

### 🧑‍💻 Contributors
```sublime
- Zagarsuren Sukhbaatar
- Diego Alejandro Ramirez Vargas
- Fendy Lomanjaya
```