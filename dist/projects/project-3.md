## 🩺 Chest X-ray Classification System

### 🔍 Project Overview:
ChestX-Ray AI is a deep learning project focused on advancing multi-label disease detection from chest radiographs. The study compares leading Convolutional Neural Networks (DenseNet121, ResNet50, EfficientNet, InceptionV3, and YOLOv11-classification) with a Swin Transformer (Swin-B) for classifying five common thoracic conditions — Atelectasis, Cardiomegaly, Effusion, Nodule, and Pneumothorax — using the NIH ChestX-ray14 dataset. The goal was to assess how well traditional CNNs’ local feature extraction and Swin Transformer’s global context modelling perform individually and as an ensemble.

The app enables users (medical professionals, researchers, or students) to:

* Upload a chest X-ray image (JPG, PNG)
* Choose from various models such as Swin Transformers, DenseNet, ResNet, EfficientNet, Inception, YOLOv11, or a custom Ensemble
* View prediction scores for conditions like **Atelectasis**, **Cardiomegaly**, **Effusion**, **Nodule**, and **Pneumothorax**
* Optionally specify the true label for performance tracking
* Visualize results via a table and bar chart

### 🧠 Supported Models
```table
| Model          | Type              | Highlights                                      |
| -------------- | ----------------- | ----------------------------------------------- |
| Swin-B         | Transformer-based | Global Attention, Hierarchical design           |
| DenseNet121    | CNN               | Efficient deep feature propagation              |
| EfficientNetB0 | CNN               | Scalable and resource-efficient                 |
| ResNet50       | CNN               | Deep residual learning                          |
| InceptionV3    | CNN               | Multi-scale convolutional feature maps          |
| YOLOv11s       | CNN (Detector)    | More detailed feature extraction                |
| Ensemble       | Hybrid            | Combines outputs of all models via voting       |
```

### ⚙️ Features

* 🖼 Upload and visualize chest X-ray images
* 🤖 Multi-model prediction
* 📊 Class probability scores
* 🧪 True label selection (for evaluation)
* 📉 Result charting for easy interpretation

### 🏥 Target Conditions

* Atelectasis
* Cardiomegaly
* Effusion
* Nodule
* Pneumothorax

### Summary of Results:
- DenseNet121 achieved the highest overall performance with an average F1-score of 0.75 and AUC > 0.91 on localised pathologies.
- Swin-B Transformer excelled at Cardiomegaly, achieving F1 = 0.80 and AUC = 0.946, due to its strong global feature reasoning.
- Model ensembling (CNNs + Swin-B) increased the average F1 score by 1%, showing complementary strengths.
- Detection performance improved over existing benchmarks: Atelectasis AUC = 0.906, Nodule AUC = 0.916, outperforming previous state-of-the-art (AUC ≈ 0.85).

### 🧪 Example Use

* Select a model (e.g., Swin-B)
* Upload a chest X-ray image
* Click **Classify**
* Review the predicted probabilities per condition
* Optionally assign a **true label** to track evaluation accuracy

![img](https://github.com/zagarsuren/chest-xray-app/blob/main/assets/ss1.jpeg?raw=true)

### 🧑‍⚕️ Medical Disclaimer

> This tool is intended for research and educational purposes only and **must not** be used for clinical diagnosis or treatment decisions.

### 🧑‍💻 Contributors
```sublime
- Zagarsuren Sukhbaatar
- Diego Alejandro Ramirez Vargas
- Shudarshan Singh Kongkham
- Dhruv Harish Punjwani
```

## 📝 License

This project is licensed under the MIT License.