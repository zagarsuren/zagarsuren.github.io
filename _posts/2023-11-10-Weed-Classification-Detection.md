# Weed classification and Detection using AI: A comparative study

## 1. Implementation Design

### 1.1. Image classification and Object detection

Image classification and object detection are two main problems of the computer vision task. While image classifiers aim to classify the image as belonging to one class, object detectors aim to find objects in an image, define the object class, and draw a bounding box around the object (Russell & Norvig, 2020).

**A)	Image classification (Backbone algorithms)**

VGG, AlexNet, GoogleNet/Inception, ResNet, and EfficientNet are renowned deep convolutional neural network architectures commonly employed backbone algorithms for image classification and diverse computer vision applications (Zaidi et al., 2022). Each of these models possesses distinct features that set them apart. 

The Visual Geometry Group introduced VGG at the University of Oxford (Simonyan & Zisserman, 2015). It is known for its simplicity and uniform architecture, consisting of a stack of convolutional layers followed by max-pooling layers. It uses small convolution filters and uses fixed parameters. 

Inception, also known as GoogLeNet, was developed by Google. It introduced the concept of "inception modules," consisting of multiple parallel convolutional layers of different kernel sizes and concatenating their outputs. The architecture is designed to capture features at multiple scales and achieve a good trade-off between model depth and computational efficiency (Szegedy et al., 2015).

ResNet, proposed by Microsoft Research, addresses the vanishing gradient problem by introducing residual or skip connections. Residual connections allow the network to learn residual functions, making it easier to train very deep networks. ResNet is known for its depth, with models like ResNet-50 and ResNet-101, which have 50 and 101 layers, respectively (He et al., 2016).

EfficientNet is a family of neural network architectures developed by Google that focuses on balancing model size, accuracy, and computational efficiency. It uses a compound scaling method to simultaneously scale the network's depth, width, and resolution. EfficientNet models are characterised by their ability to achieve competitive performance with fewer parameters, making them suitable for resource-constrained environments. The family includes different model variants from EfficientNet B0 to B7 (Tan & Le, 2020).

![Imgur](https://i.imgur.com/vHZNFPJ.png)
*Figure 1. Model scaling of EfficientNet (Tan & Le, 2020)*

In summary, VGG is renowned for its simple architecture, Inception places importance on capturing features across various scales, ResNet introduces skip connections to enable the creation of extremely deep networks, and EfficientNet prioritises optimising the trade-off between performance and computational efficiency.

Performance comparison on ImageNet dataset. EfficientNet reached a significantly better accuracy with fewer parameters than other model architectures.

![Imgur](https://i.imgur.com/oB0Souq.png)
*Figure 2. Performance comparison of different model sizes vs. ImageNet Accuracy (Tan & Le, 2020)*

We will employ the EfficientNet model in our weed classification problem by comparing the backbone algorithms' performance and efficiency. 

**B)	Object detectors**

Object detection is a fundamental computer vision problem that involves identifying and localising objects of interest within an image or a video stream. The goal of object detection is to not only classify the objects present in the scene but also to determine their precise locations by drawing bounding boxes around them. This task is crucial in various applications, including autonomous vehicles, surveillance systems, medical imaging, and robotics. 

Object detection can be approached using various techniques, including traditional computer vision and deep learning-based methods. Deep learning techniques, especially Convolutional Neural Networks (CNNs), have significantly improved object detection performance in recent years. 

Single-stage and two-stage object detectors are two different approaches to solving the problem of object detection in computer vision. They differ in their architecture and how they process images to identify and locate objects within them. YOLO (You Only Look Once), SSD (Single Shot MultiBox Detector), RetinaNet, and EfficientDet are examples of single-stage detectors and R-CNN, Fast R-CNN, and Faster R-CNN are examples of two-stage detectors. 


**Single-stage detectors**

YOLO is renowned for its speed and ability to detect objects in real-time. It accomplishes this by conducting object detection in a single neural network pass, directly forecasting bounding boxes and class probabilities. YOLO subdivides the image into a grid and designates bounding boxes to grid cells that contain objects. Despite its speed, YOLO may encounter challenges when detecting small objects and achieving precise localisation (Redmon et al., 2016). The example of YOLO object detection is illustrated in the figure below.

![Imgur](https://i.imgur.com/PIdnNlX.png)
*Figure 3. The YOLO model (Redmon et al., 2016)*

SSD was created to tackle the issue of detecting objects across various scales. It forecasts bounding boxes and class labels on multiple feature maps with differing resolutions. This approach at different scales empowers SSD to handle objects of varying sizes effectively. It balances speed and accuracy, making it well-suited for real-time applications (Liu et al., 2016).

RetinaNet brought about the Focal Loss, which effectively tackles the issue of class imbalance in object detection, arising from the abundance of background (non-object) regions compared to object regions. It combines a one-stage object detection method with exceptional accuracy. RetinaNet employs a Feature Pyramid Network (FPN) to detect objects across various scales. It is celebrated for its proficiency in handling small objects and maintaining a robust balance between speed and precision (Lin et al., 2018).

EfficientDet is a series of object detection models that enhance model architecture and efficiency through a technique known as compound scaling. It merges the efficiency of EfficientNet with object detection techniques. EfficientDet is renowned for achieving competitive performance using fewer parameters, rendering it well-suited for environments with limited computational resources (Tan et al., 2020).

**Two-stage detectors**

Region-based Convolutional Neural Network (R-CNN), Fast R-CNN, Faster R-CNN, and Mask R-CNN are object detection models that have developed progressively. Each model enhances its predecessors, tackling shortcomings and enhancing precision and computational efficiency.

The R-CNN showcased the substantial enhancement in detection performance achievable through CNNs. R-CNN employs a region proposal module that is class-agnostic and utilizes CNNs to transform the task of detection into a problem of classification and localization (Zaidi et al., 2022).

![Imgur](https://i.imgur.com/zB84HHT.png)
*Figure 4. R-CNN (Zaidi et al., 2022)*

A drawback of R-CNN was the requirement to train multiple systems independently. Fast R-CNN addressed this by introducing a single, end-to-end trainable system. In this system, an image and object proposals are fed as input. The image undergoes convolution layers to generate feature maps, while the object proposals are aligned with these maps.

### 1.2. Comparison of AI techniques
The pros and cons of single-stage and two-stage detectors are summarised in the figure below.

|	|Pros|	Cons|
|---|-----|-------|
|Single-stage detectors	| Single Pass Detection. One-stage detectors perform object detection in a single pass through the neural network. They predict object bounding boxes and class labels directly in one step. Faster Inference. One-stage detectors tend to be faster during inference because they do not require the additional step of generating region proposals. Simplicity. Single-stage detectors are often more straightforward regarding architecture and have fewer components than two-stage detectors.|	Accuracy. One-stage detectors can achieve good results for many tasks but may have slightly lower accuracy than two-stage detectors. The single-stage detectors might need help precisely localising objects, especially when objects are small or densely packed. |
|Two-stage detectors|	Region Proposal Network (RPN). Two-stage detectors have a two-step process. The first stage is dedicated to generating region proposals, which are areas of interest likely to contain objects. This is typically done using an RPN, which produces a set of candidate object proposals.Refinement. In the second stage, these region proposals are refined by another neural network to predict the precise object location and class label.Better Localization. Two-stage detectors often have better object localization accuracy, especially for small or densely packed objects, as the region proposals help narrow the search space.	| Complexity. Two-stage detectors tend to be more complex in terms of architecture due to the two-step process. Slower Inference. Two-stage detectors usually have slower inference times because of the additional region proposal step.|

In object detection tasks, researchers and engineers typically opt for detectors that align most effectively with the particular trade-offs among accuracy, speed, and resource limitations that their specific applications demand. One-stage detectors tend to be the preferred option when prioritising real-time processing and faster inference. On the other hand, if the emphasis is on achieving high accuracy and pinpointing object locations precisely, two-stage detectors may be the more favourable choice.

Zaidi et al (2022), compared the different object detectors on MS COCO and PASCAL VOC 2012 datasets.  The figure below compares the average precision performance on MS COCO dataset. The result showed the best models were Swin-L, DetectroRS and YOLO v5 and v6.

![Imgur](https://i.imgur.com/DuECAbE.png)
*Figure 5. Performance of object detectors on MS COCO dataset. Adapted from (Zaidi et al., 2022)*

**YOLO on weed detection problem**

Recent research (Dang et al., 2023) showed the benchmark of YOLO object detectors in the cotton weed dataset. YOLO models showed promising training performance regarding fast convergence and high detection accuracies. 

![Imgur](https://i.imgur.com/Dzviqjg.png)
*Figure 6. Training curves of top 10 YOLO models for weed detection. (Dang et al., 2023)*

**Mean average precision**
Scaled-YOLOv4 achieved the highest mean average precision (mAP) of 95.03% and 89.53%, respectively. YOLOv4 and YOLOR also demonstrated high accuracy, closely following Scaled-YOLOv4. They were succeeded by YOLOv6, YOLOV7, YOLOv5, and YOLOv3 in terms of their performance. Notably, detection models based on YOLOv4 outperformed those based on YOLOv5 and YOLOv3, particularly in the context of weed detection accuracy (Dang et al., 2023).

**Inference time**
Of the seven variations of YOLO models, YOLOv5 models, particularly the simplified versions YOLOv5n and YOLOv5s, excelled in computational efficiency, being the fastest in terms of inference time, requiring less than 2 milliseconds (Dang et al., 2023).

The results above show that the YOLO can be used in our project's weed classification and detection tasks. 

### 1.3. Data structure
Across all precision agriculture tasks that rely on computer vision, a shared objective appears to be identifying specific objects like crops, weeds, or fruits within the agricultural setting and distinguishing them from the surrounding environment (Lu & Young, 2020). Achieving this goal necessitates a well-structured hardware system and a robust data analysis process, typically involving training deep learning models using specialised image datasets. MS COCO (Microsoft Common Objects in Context) format creates the image classification and object detection tasks. It is a JSON file structure including the images, labels, and bounding boxes. A detailed structure of the COCO format is displayed in the Appendices section. 

**A. Weed classification dataset**

DeepWeeds dataset (Olsen et al., 2019). The dataset designed for deep learning focused on classifying various weed species in a multiclass setting. It consists of images captured in rangeland pastures, each categorised by a specific weed species. The figure below shows the sample images of various weed species.

![Imgur](https://i.imgur.com/8KSptQX.png)
*Figure 7. Sample images from each class of the DeepWeeds dataset, namely: (a) Chinee apple, (b) Lantana, (c) Parkinsonia, (d) Parthenium, (e) Prickly acacia, (f) Rubber vine, (g) Siam weed, (h) Snake weed and (i) Negatives. (Olsen et al., 2019)* 

**B. Weed detection datasets**

CottonWeedDet12 dataset (Dang et al., 2023). The CottonWeedDet12 dataset comprises 5,648 RGB images showcasing 12 weed species commonly found in cotton fields across the southern United States. It includes a total of 9,370 bounding boxes. These images were captured between June and September 2021 using smartphones or handheld digital cameras in natural field lighting conditions. Qualified individuals conducted the manual labelling for weed identification using the VGG Image Annotator.

![Imgur](https://i.imgur.com/4YtEliO.png)
*Figure 8. Sample images and annotations of CottonWeedDet dataset (Dang et al., 2023)*

### 1.4. Implementation design 

**A. Weed classification** 

**Data preparation and augmentation**

Data augmentation is a popular and easy method to improve a classifier's generalisation by increasing the training set’s size by adding extra copies of the training examples that have been modified with transformations (Goodfellow et al., 2016). We will use the Albumentation library in the data augmentation (Buslaev et al., 2020). 

Some of the transformations of Albumentations include, RandomResizedCrop, Transpose, HorizontalFlip, VerticalFlip, ShiftScaleRotate, HueSaturationValue, RandomBrightnessContrast, Normalize, CoarseDropout, and Cutout. More detailed settings of augmentation can be seen from appendices section. 

Some examples of data augmentation on YOLOWeeds dataset shown in the figure below.

![Imgur](https://i.imgur.com/XczZ7Ej.png)
*Figure 9. Data augmentation examples of YOLOWeeds (Dang et al., 2023)*

**EfficientNet and Transfer learning**

We will use the Pytorch framework and EfficientNet-PyTorch (https://github.com/lukemelas/EfficientNet-PyTorch) library to train the EfficientNet model. 

```python
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')
```
Required libraries and utility functions: torch, round_filters, round_repeats, drop_connect, get_same_padding_conv2d, get_model_params, efficientnet_params, load_pretrained_weights, Swish, MemoryEfficientSwish, calculate_output_image_size.

```Python
VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7'
)
```

The EfficientNet class employ the following methods:
- extract_features(inputs): It takes the input tensor and returns the final convolutional layer in the efficientnet model.
- forward(inputs): It is an efficientnet forward function. It calls extract features, applies the final linear layer and returns the logits layer. 
- from_pretrained: This method uses different arguments as input, such as model name, a path to pre-trained weights, the number of classes – and the number of categories for classification. It also controls the output size for the final linear layer and returns the pretrained efficientnet model. 
For efficientnet-b4 model, the output layer uses 1792 units; for the b7 model, it uses 2560 units.

Pretrained weights:

```Python 
EfficientNet.from_pretrained("efficientnet-b4")
```

Output layer for custom model:
```Python
self.out = nn.Linear(1792, num_classes)
```

**B. Weed detection - YOLO**

The YOLOv5 implementation within the PyTorch framework is continuously progressing, offers versatile control over the model's dimensions and is compatible with various devices for the execution (Dang et al., 2023). We will use the ultralytics library to train our YOLO weed detector (https://pytorch.org/hub/ultralytics_yolov5/). 

![Imgur](https://i.imgur.com/Gl6fQQ3.png)
*Figure 10. YOLO model types*

**1. YOLOv5 Transfer Learning preparation**
Required libraries `torch==1.10.1, torchvision==0.11.2.`

We will prepare our datasets and classes in the Yaml file structure.

```yaml
# Dataset paths relative to the yolov5 folder
train: ../data/images/train
val:   ../data/images/val
test:  ../data/images/test

# Number of classes
nc: Number of weed classes

# Class names 
names: ['class1', 'class2', …]
```

**2. Freeze the YOLOv5 Backbone**
The backbone refers to the layers responsible for extracting features from the input image. In the context of YOLOv5 transfer learning, we intend to keep the backbone layers fixed, preventing weight updates. Instead, our focus will solely be training the final layers, known as the "head layers." 

**3. Training**
We will use the following command to execute the model. 

```terminal
python yolov5/train.py --data weeds.yaml --weights yolov5s.pt --epochs 100 --batch 4 --freeze 10
```

Arguments:
- data: the dataset definition YAML file
- weights: the pre-trained YOLOv5 model weights.  
- epochs: the number of epochs = 100 
- batch: the batch size
- freeze: the number of layers to freeze.

The best model weights will be generated after training the model on the data.

**4. Performance evaluation**
The following image is adapted from Dang et al. to illustrate the prediction result. 

![Imgur](https://i.imgur.com/UQ0EQTQ.png)
*Figure 11. Example prediction of YOLO model (Dang et al., 2023)*

The model's performance employs different evaluations, and the following section will explain the metrics. 

# 2. Evaluation Design

**Evaluation metrics**

Object detectors employ several criteria to assess their performance, including metrics such as frames per second (FPS), precision, and recall. Nevertheless, the mean Average Precision (mAP) is the most prevalent evaluation measure. Precision is calculated based on Intersection over Union (IoU), which quantifies the ratio of the overlapping area to the union area between the actual object and the predicted bounding box. An established threshold is utilised to determine the accuracy of the detection. When the IoU surpasses this threshold, it is classified as a True Positive, whereas an IoU below the threshold is considered a False Positive. If the model fails to detect an object that exists in the ground truth, it is termed a False Negative. Precision assesses the percentage of accurate predictions, whereas recall evaluates the accurate predictions relative to the ground truth (Zaidi et al., 2022).

$Precision =  {True Positives \over True Positives+False Positives}$

$Recall =  {True Positives \over True Positives+False Negatives}$


**mAP – Mean average precision**

The standard description of Average Precision (AP) involves calculating the area under the precision-recall curve.

![Imgur](https://i.imgur.com/4IDaXi2.png)
*Figure 12. Precision-Recall curve*

To calculate the mAP, average the AP values for all object categories. This comprehensively assesses the model's performance in detecting objects across various classes. mAP is a valuable metric for comparing and ranking different object detection models. A higher mAP score signifies a model with superior accuracy in detecting objects across diverse categories. It proves especially beneficial when dealing with datasets with imbalanced class distributions or evaluating a model's performance in a multi-class detection context.

**IoU – Intersection over Union**

![Imgur](https://i.imgur.com/aAqHO4O.png)
*Figure 13. Union and intersection*

Intersection over union is a measurement used to evaluate the accuracy of object detection and localisation. IoU calculates the area of overlap between the predicted bounding box and the ground truth bounding box.
Union calculates the area encompassed by the predicted bounding box and the ground truth bounding box, including their areas of overlap. To compute IoU, divide the area of intersection by the area of union. Mathematically, it's represented as:


$$IoU =  {AreaofIntersection \over Area of Union}$$

---

## List of References
Buslaev, A., Parinov, A., Khvedchenya, E., Iglovikov, V. I., & Kalinin, A. A. (2020). Albumentations: Fast and flexible image augmentations. Information, 11(2), 125. https://doi.org/10.3390/info11020125

Dang, F., Chen, D., Lu, Y., & Li, Z. (2023). YOLOWeeds: A novel benchmark of YOLO object detectors for multi-class weed detection in cotton production systems. Computers and Electronics in Agriculture, 205, 107655. https://doi.org/10.1016/j.compag.2023.107655

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. http://www.deeplearningbook.org

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778. https://doi.org/10.1109/CVPR.2016.90

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2018). Focal Loss for Dense Object Detection (arXiv:1708.02002). arXiv. http://arxiv.org/abs/1708.02002

Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.-Y., & Berg, A. C. (2016). SSD: Single Shot MultiBox Detector (Vol. 9905, pp. 21–37). https://doi.org/10.1007/978-3-319-46448-0_2

Lu, Y., & Young, S. (2020). A survey of public datasets for computer vision tasks in precision agriculture. Computers and Electronics in Agriculture, 178, 105760. https://doi.org/10.1016/j.compag.2020.105760

Olsen, A., Konovalov, D. A., Philippa, B., Ridd, P., Wood, J. C., Johns, J., Banks, W., Girgenti, B., Kenny, O., Whinney, J., Calvert, B., Azghadi, M. R., & White, R. D. (2019). DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning. Scientific Reports, 9(1), 2058. https://doi.org/10.1038/s41598-018-38343-3

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection (arXiv:1506.02640). arXiv. http://arxiv.org/abs/1506.02640

Russell, S. J., & Norvig, P. (2020). Artificial intelligence: A modern approach (4th edition). Pearson Education Limited.

Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition (arXiv:1409.1556). arXiv. http://arxiv.org/abs/1409.1556

Szegedy, C., Wei Liu, Yangqing Jia, Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., & Rabinovich, A. (2015). Going deeper with convolutions. 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 1–9. https://doi.org/10.1109/CVPR.2015.7298594

Tan, M., & Le, Q. V. (2020). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (arXiv:1905.11946). arXiv. http://arxiv.org/abs/1905.11946

Tan, M., Pang, R., & Le, Q. V. (2020). EfficientDet: Scalable and Efficient Object Detection (arXiv:1911.09070). arXiv. http://arxiv.org/abs/1911.09070

Zaidi, S. S. A., Ansari, M. S., Aslam, A., Kanwal, N., Asghar, M., & Lee, B. (2022). A survey of modern deep learning based object detection models. Digital Signal Processing, 126, 103514. https://doi.org/10.1016/j.dsp.2022.103514