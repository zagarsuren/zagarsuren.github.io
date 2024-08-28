# HerdWatch: Livestock Monitoring and Behaviour Analysis using Computer Vision (Project)

## Abstract
Livestock management is a crucial aspect of agriculture, requiring efficient methods for counting and monitoring animal behaviour. Traditional methods rely heavily on manual labour and can be time-consuming, costly, and prone to errors. In recent years, computer vision techniques have emerged as a promising solution to automate these processes. This project aims to develop a system for livestock counting and behaviour detection using computer vision algorithms. The proposed system utilises state-of-the-art deep learning techniques to process images or video footage captured from surveillance cameras installed in livestock facilities or drones. In this report, we developed a cattle behaviour detection system and compared the results using Faster-RCNN, YOLOv5, YOLOv8 nano and YOLOv8 medium object detection models. 
## 1. Introduction and Background 
This project aims to enhance livestock management by developing a computer vision-based system for automated cattle counting and behaviour monitoring. Motivated by the need to detect and mitigate cattle diseases, the system provides real-time insights into cattle health and behaviour, enabling early intervention and better farm productivity. Applications include health monitoring, reproductive and nutritional management, productivity enhancement, and predictive analytics, all contributing to improved animal welfare and farm sustainability. 
### 1.1 The problem 
The problem we aimed to solve is the inefficiency and limitations of traditional methods for livestock counting and behaviour monitoring in agriculture. Manual counting and observation are labour-intensive, time-consuming, and prone to errors, leading to resource allocation inefficiencies and potential animal welfare issues. By developing a computer vision-based system, we sought to automate these tasks, improving accuracy, speed, and reliability while reducing the burden on farmers and enhancing overall livestock health and management practices.
### 1.2 Motivation
Detecting and mitigating cattle diseases is crucial for ensuring the health and welfare of livestock, as well as maintaining farm productivity. Infectious diseases, such as Bovine pestivirus or Bovine viral diarrhoea (BVD), pose significant threats to cattle populations and can result in severe economic losses. In cattle in Australia and New Zealand, BVD primarily affects the developing fetus and is a major contributor to reproductive losses, leading to conception failures, genetic abnormalities, and deaths (Kirkland & MacKintosh, 2016). These risks underscore the critical role of management practices in maintaining animal well-being. Implementing an AI-based cattle detection and monitoring system addresses these challenges by providing real-time insights into cattle behaviour. By analyzing cattle behaviours, the AI system can detect early signs of illness, enabling timely intervention to prevent disease spread. This project aims to enhance animal welfare, farm sustainability, and economic viability in the livestock industry through the help of cutting-edge computer vision technology. 

### 1.3 Dataset
We utilized the cow behaviour detection (Version 1) dataset (CowBehaviorDetection, 2024) from Roboflow in our project. It consists of 1215 cow images, bounding boxes and their labels. The six behaviour classes include estrus, grazing, lying, smelling, standing and null. The sample images are displayed in the figure below.

![img](/assests/livestock_project/cattle1.png)

**Data preparation and cleaning process**

We noticed some wrong annotations in the original dataset during the initial experiment. Therefore, we uploaded all the training, validation, test images and annotations to CVAT.ai and checked and fixed all annotations. In addition, because of very few samples and confusion, we removed the “smelling” and “Null” categories from the dataset. After cleaning data, the four behaviour classes (Estrus, grazing, lying, and standing) are used in the final model training. We also added some images to the train set with their annotations.
The annotation example is shown in Figure below.

![img](/assests/livestock_project/cattle2.png)

**Dataset split details**

The dataset is split into training (70%), validation (10%), and test sets (20%). The training dataset was used to train our model, and the validation set was used to monitor the model’s performance during training. After data preparation, the total number of samples was 1229 images and the number of samples in each set outlined below.
* Train set: 857
* Validation set: 247
* Test set: 125

## 2. GUI design
We developed web application using Django framework. 
![img](/assests/livestock_project/cattle11.png)

## 3. Results and Evaluation

### 3.1 Experimental Settings
We trained models on our final dataset, consisting of 1229 images. The behaviour classes include estrus, grazing, lying, and standing. We trained Faster R-CNN (Ren et al., 2016), YOLO (Redmon et al., 2016) versions v5 small, v8 nano, and v8 medium models. The experimental settings and results are provided in the following sections. 

**Model 1. Faster R-CNN** 

Model summary:
* Total params: 43,271,528
* Trainable params: 43,046,184
* Non-trainable params: 225,344
* Total mult-adds (G): 559.92

`Classes: ['__background__', 'Estrus', ‘grazing', 'lying', 'standing’]`

Training configurations:

* Model: fasterrcnn_resnet50_fpn_v2
* Number of Epochs: 50
* Batch size = 2

Training time: 4 hours
 
**Model 2. YOLOv5 small**

Train_yaml summary: 
* 182 layers
* 7,254,609 parameters
* 0 gradients
Training configurations:
* weights=yolov5s.pt
* image size = 416
* batch size = 16
* Epochs = 200

hyperparameters: `lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0`

Training time: 0.799 hours


**Model 3. YOLOv8 nano** 

Model summary:
* 168 layers
* 3,006,428 parameters
* 0 gradient
* 8.1 GFLOPs

Train configurations:
* model = YOLO("yolov8n.pt")
* imgsz = 640,
* epochs = 500,
* batch = -1,
* name = 'yolov8n_final',

Training time: 0.78 hours

Both batch size and optimizer are automatically determined by the model.
* Batch size: 71
* Optimizer: AdamW(lr=0.001111, momentum=0.9)

Model 4. YOLOv8 medium 

Model summary:
* 218 layers
* 25,842,076 parameters
* 0 gradients
* 78.7 GFLOPs

Train configurations:
* model = yolov8m.pt
* imgsz=640,
* epochs=500,
* batch=-1,
* name='yolov8m_final',

Training time: 0.84 hours

Both batch size and optimizer are automatically determined by the model.
-	Batch size: 20
-	Optimizer: AdamW(lr=0.001111, momentum=0.9)

 

### 3.2 Experimental Results 
The table provides performance metrics for four models regarding mean Average Precision (mAP) at different Intersection over Union (IoU) thresholds.

mAP 0.5 and mAP 0.5:0.95
![img](/assests/livestock_project/cattle3.png)

Validation results by class:

Average Precision
![img](/assests/livestock_project/cattle4.png)

Average Recall
![img](/assests/livestock_project/cattle5.png)

YOLOv8 medium outperformed the other architectures regarding mAP across both training and validation datasets, indicating its effectiveness in this object detection task.

These results show that the model performance was improved by around 7 to 8% compared to our initial experiments (Initial experiments mAP:0.5 on validation set: Faster RCNN 0.805, YOLOv8 Nano 0.805, YOLOv8 Medium 0.836). This indicates the importance of the data cleaning and preparation process.  

**Training plots**
Faster R-CNN
![img](/assests/livestock_project/cattle6.png)

YOLOv5 small
![img](/assests/livestock_project/cattle7.png)

YOLOv8 nano
![img](/assests/livestock_project/cattle8.png)

YOLOv8 medium
![img](/assests/livestock_project/cattle9.png)

Inference results
![img](/assests/livestock_project/cattle10.png)

### Demo Video:
<iframe width="850" height="540" src="https://www.youtube.com/embed/E_jsOMbI-yE?si=AgPci3GnaYb6quON" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### 3.4 Limitations

Small object detection and occlusions: We tested our model on drone footage and noticed a few limitations, such as small object detection and occlusion issues. The system cannot detect small objects since we trained our model on medium and large cattle images. In addition, the system returns the wrong behaviour classes if heavy occlusions exist. The sample images are depicted in the below images.

## 4. Discussion and Conclusions 
Our project aimed to develop and evaluate object detection models for classifying cattle behaviours. The objective was to create a system that accurately identifies estrus, grazing, lying, and standing behaviours to enhance livestock management practices and farm productivity.

We experimented with four prominent object detection architectures: Faster R-CNN, YOLOv5 small, YOLOv8 nano, and YOLOv8 medium. Each model was trained on the final cattle image dataset and was evaluated on both training and validation datasets.

Performance Evaluation:
* Faster R-CNN: Achieved a validation mAP of 0.891 at IoU 0.5.
* YOLOv5 small: Obtained a validation mAP of 0.855 at IoU 0.5.
* YOLOv8 nano: Achieved a validation mAP of 0.879 at IoU 0.5.
* YOLOv8 medium: Demonstrated the highest performance, with a validation mAP of 0.906 at IoU 0.5.

Our project identified two primary limitations:
* Small Object Detection. Due to training on medium and large cattle images, models struggled to detect small objects in drone footage.
* Occlusion Issues. Heavy occlusions led to incorrect behaviour class predictions, affecting the system’s accuracy.
Future extensions and improvements: 
* Fine-tuning the models using a diverse dataset focused on small cattle or objects to improve small object detection.
* Due to the lack of data, the solution was limited to only detecting 4 out of the many behaviours that cattle can exhibit. Therefore, the model can be further improved by training on a dataset with more behaviours.
* The model currently being used was trained on a dataset of cattle behaviours and not cattle detection. Therefore, reducing its effectiveness in cattle counting and using one model for cattle detection and another for behaviour classification might alleviate this problem.
* Extending the model on the cattle disease dataset could be useful for detecting cattle diseases in a more accurate way.

**Contributors:**
- Zagarsuren Sukhbaatar
- Hoang Bao Nguyen 
- Yi Yang

---
## 5. References 

CowBehaviorDetection. (2024). Cow behavior detection dataset [Open Source Dataset]. Roboflow Universe. [https://universe.roboflow.com/cowbehaviordetection/cow_behavior_detection-veqgv](https://universe.roboflow.com/cowbehaviordetection/cow_behavior_detection-veqgv) <br><br>
Jacob, S., & Francesco. (2024). What is YOLOv8? The Ultimate Guide. [https://blog.roboflow.com/whats-new-in-yolov8/](https://blog.roboflow.com/whats-new-in-yolov8/)<br><br>
Kirkland, P., & MacKintosh, S. (2016). Ruminant Pestivirus Infections. Australia and New Zealand Standard Diagnostic Procedures. [https://www.agriculture.gov.au/sites/default/files/sitecollectiondocuments/animal/ahl/ANZSDP-Pestiviruses.pdf](https://www.agriculture.gov.au/sites/default/files/sitecollectiondocuments/animal/ahl/ANZSDP-Pestiviruses.pdf)<br><br>
Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection (arXiv:1506.02640). arXiv. [http://arxiv.org/abs/1506.02640](http://arxiv.org/abs/1506.02640)<br><br>
Ren, S., He, K., Girshick, R., & Sun, J. (2016). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (arXiv:1506.01497). arXiv. [http://arxiv.org/abs/1506.01497](http://arxiv.org/abs/1506.01497)


