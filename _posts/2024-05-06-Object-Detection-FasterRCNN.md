# Object Detection Faster-RCNN (Code) <br/>

# Outline
1. Environment preparation
2. Data Preparation
3. Model Training
4. Results and Evaluation

# 1. Environment Preparation




```python
# Clone the Faster RCNN model repository
!git clone https://github.com/ObjectDetection-FasterRCNN.git
```

    Cloning into 'ObjectDetection-FasterRCNN'...
    remote: Enumerating objects: 127, done.[K
    remote: Counting objects: 100% (127/127), done.[K
    remote: Compressing objects: 100% (66/66), done.[K
    remote: Total 127 (delta 60), reused 123 (delta 58), pack-reused 0[K
    Receiving objects: 100% (127/127), 4.26 MiB | 20.50 MiB/s, done.
    Resolving deltas: 100% (60/60), done.
    Updating files: 100% (68/68), done.



```python
# Change the directory to the FasterRCNN repo
%cd /root/ObjectDetection-FasterRCNN/
```

    /root/ObjectDetection-FasterRCNN



```python
# Install the Requirements
!pip install -r requirements.txt
```


# 2. Data Preparation

Copy Pascal VOC dataset to the FasterRCNN folder


```python
# Copy Pascal VOC dataset to FasterRCNN/data folder
!cp -r /root/data/Object_Detection/pascal /root/ObjectDetection-FasterRCNN/data
```


```python
# Display images
import os
import cv2
import random
import matplotlib.pyplot as plt

# Function to display images from a folder
def display_images(folder_path, num_images=10):
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.ravel()

    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random_image_files = random.sample(image_files, num_images)

    for i, image_file in enumerate(random_image_files):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axs[i].imshow(image)
        axs[i].axis('off')
        axs[i].set_title(image_file)
        print(image.shape)
    plt.show()

# Path to the folder 
folder_path = "/root/ObjectDetection-FasterRCNN/data/pascal/train"

# Display sample images 
display_images(folder_path)
```

    
![png](/assests/faster_rcnn_lettuce/output_10_1.png)
    


Prepare an YAML file


```python
%%writefile data_configs/custom_data.yaml
# Define train and validation directory
TRAIN_DIR_IMAGES: '/root/ObjectDetection-FasterRCNN/data/pascal/train'
TRAIN_DIR_LABELS: '/root/ObjectDetection-FasterRCNN/data/pascal/train'
VALID_DIR_IMAGES: '/root/ObjectDetection-FasterRCNN/data/pascal/valid'
VALID_DIR_LABELS: '/root/ObjectDetection-FasterRCNN/data/pascal/valid'

# Class names including background
CLASSES: [
    '__background__',
    'Ready', 
    'empty_pod', 
    'germination', 
    'pod', 
    'young'
]

# Number of classes (object classes + 1 for background class in Faster RCNN).
NC: 6

# Whether to save the predictions of the validation set while training.
SAVE_VALID_PREDICTION_IMAGES: True
```

    Writing data_configs/custom_data.yaml



# 3. Model Training

Create `outputs`, `training`, `plantdetection` directories.


```python
!mkdir /root/ObjectDetection-FasterRCNN/outputs
!mkdir /root/ObjectDetection-FasterRCNN/outputs/training
!mkdir /root/ObjectDetection-FasterRCNN/outputs/training/plantdetection
```

Training configurations:
- data configs: custom_data.yaml
- epochs: 25
- backbone model: fasterrcnn_resnet50_fpn_v2
- batch size: 2
- no mosaic


```python
!python train.py --config data_configs/custom_data.yaml --epochs 25 --model fasterrcnn_resnet50_fpn_v2 --project-name plantdetection --batch-size 2 --no-mosaic
```

    Not using distributed mode
    device cuda
    Creating data loaders
    Number of training samples: 1057
    Number of validation samples: 226
    
    Building model from scratch...
    ====================================================================================================
    Layer (type:depth-idx)                             Output Shape              Param #
    ====================================================================================================
    FasterRCNN                                         [100, 4]                  --
    ├─GeneralizedRCNNTransform: 1-1                    [2, 3, 800, 800]          --
    ├─BackboneWithFPN: 1-2                             [2, 256, 13, 13]          --
    │    └─IntermediateLayerGetter: 2-1                [2, 2048, 25, 25]         --
    │    │    └─Conv2d: 3-1                            [2, 64, 400, 400]         (9,408)
    │    │    └─BatchNorm2d: 3-2                       [2, 64, 400, 400]         (128)
    │    │    └─ReLU: 3-3                              [2, 64, 400, 400]         --
    │    │    └─MaxPool2d: 3-4                         [2, 64, 200, 200]         --
    │    │    └─Sequential: 3-5                        [2, 256, 200, 200]        (215,808)
    │    │    └─Sequential: 3-6                        [2, 512, 100, 100]        1,219,584
    │    │    └─Sequential: 3-7                        [2, 1024, 50, 50]         7,098,368
    │    │    └─Sequential: 3-8                        [2, 2048, 25, 25]         14,964,736
    │    └─FeaturePyramidNetwork: 2-2                  [2, 256, 13, 13]          --
    │    │    └─ModuleList: 3-15                       --                        (recursive)
    │    │    └─ModuleList: 3-16                       --                        (recursive)
    │    │    └─ModuleList: 3-15                       --                        (recursive)
    │    │    └─ModuleList: 3-16                       --                        (recursive)
    │    │    └─ModuleList: 3-15                       --                        (recursive)
    │    │    └─ModuleList: 3-16                       --                        (recursive)
    │    │    └─ModuleList: 3-15                       --                        (recursive)
    │    │    └─ModuleList: 3-16                       --                        (recursive)
    │    │    └─LastLevelMaxPool: 3-17                 [2, 256, 200, 200]        --
    ├─RegionProposalNetwork: 1-3                       [1000, 4]                 --
    │    └─RPNHead: 2-3                                [2, 3, 200, 200]          --
    │    │    └─Sequential: 3-18                       [2, 256, 200, 200]        1,180,160
    │    │    └─Conv2d: 3-19                           [2, 3, 200, 200]          771
    │    │    └─Conv2d: 3-20                           [2, 12, 200, 200]         3,084
    │    │    └─Sequential: 3-21                       [2, 256, 100, 100]        (recursive)
    │    │    └─Conv2d: 3-22                           [2, 3, 100, 100]          (recursive)
    │    │    └─Conv2d: 3-23                           [2, 12, 100, 100]         (recursive)
    │    │    └─Sequential: 3-24                       [2, 256, 50, 50]          (recursive)
    │    │    └─Conv2d: 3-25                           [2, 3, 50, 50]            (recursive)
    │    │    └─Conv2d: 3-26                           [2, 12, 50, 50]           (recursive)
    │    │    └─Sequential: 3-27                       [2, 256, 25, 25]          (recursive)
    │    │    └─Conv2d: 3-28                           [2, 3, 25, 25]            (recursive)
    │    │    └─Conv2d: 3-29                           [2, 12, 25, 25]           (recursive)
    │    │    └─Sequential: 3-30                       [2, 256, 13, 13]          (recursive)
    │    │    └─Conv2d: 3-31                           [2, 3, 13, 13]            (recursive)
    │    │    └─Conv2d: 3-32                           [2, 12, 13, 13]           (recursive)
    │    └─AnchorGenerator: 2-4                        [159882, 4]               --
    ├─RoIHeads: 1-4                                    [100, 4]                  --
    │    └─MultiScaleRoIAlign: 2-5                     [2000, 256, 7, 7]         --
    │    └─FastRCNNConvFCHead: 2-6                     [2000, 1024]              --
    │    │    └─Conv2dNormActivation: 3-33             [2000, 256, 7, 7]         590,336
    │    │    └─Conv2dNormActivation: 3-34             [2000, 256, 7, 7]         590,336
    │    │    └─Conv2dNormActivation: 3-35             [2000, 256, 7, 7]         590,336
    │    │    └─Conv2dNormActivation: 3-36             [2000, 256, 7, 7]         590,336
    │    │    └─Flatten: 3-37                          [2000, 12544]             --
    │    │    └─Linear: 3-38                           [2000, 1024]              12,846,080
    │    │    └─ReLU: 3-39                             [2000, 1024]              --
    │    └─FastRCNNPredictor: 2-7                      [2000, 6]                 --
    │    │    └─Linear: 3-40                           [2000, 6]                 6,150
    │    │    └─Linear: 3-41                           [2000, 24]                24,600
    ====================================================================================================
    Total params: 43,276,653
    Trainable params: 43,051,309
    Non-trainable params: 225,344
    Total mult-adds (G): 559.93
    ====================================================================================================
    Input size (MB): 9.83
    Forward/backward pass size (MB): 7478.59
    Params size (MB): 173.11
    Estimated Total Size (MB): 7661.53
    ====================================================================================================
    43,276,653 total parameters.
    43,051,309 training parameters.
    Epoch: [0]  [  0/529]  eta: 0:15:22  lr: 0.000003  loss: 3.1651 (3.1651)  loss_classifier: 1.9201 (1.9201)  loss_box_reg: 0.8446 (0.8446)  loss_objectness: 0.3180 (0.3180)  
    Epoch: [24]  [  0/529]  eta: 0:09:39  lr: 0.001000  loss: 0.0666 (0.0666)  loss_classifier: 0.0246 (0.0246)  loss_box_reg: 0.0406 (0.0406)  loss_objectness: 0.0004 (0.0004)  loss_rpn_box_reg: 0.0009 (0.0009)  time: 1.0945  data: 0.3007  max mem: 4143
    Epoch: [24]  [100/529]  eta: 0:05:07  lr: 0.001000  loss: 0.0993 (0.0923)  loss_classifier: 0.0323 (0.0323)  loss_box_reg: 0.0603 (0.0572)  loss_objectness: 0.0002 (0.0004)  loss_rpn_box_reg: 0.0025 (0.0024)  time: 0.7136  data: 0.0022  max mem: 4143
    Epoch: [24]  [200/529]  eta: 0:03:55  lr: 0.001000  loss: 0.0878 (0.0892)  loss_classifier: 0.0312 (0.0313)  loss_box_reg: 0.0536 (0.0552)  loss_objectness: 0.0002 (0.0004)  loss_rpn_box_reg: 0.0019 (0.0023)  time: 0.7143  data: 0.0021  max mem: 4143
    Removed invalid box tensor([207, 107, 207, 114]) of 3 from image index 510
    Epoch: [24]  [300/529]  eta: 0:02:43  lr: 0.001000  loss: 0.0874 (0.0890)  loss_classifier: 0.0283 (0.0313)  loss_box_reg: 0.0551 (0.0551)  loss_objectness: 0.0002 (0.0004)  loss_rpn_box_reg: 0.0020 (0.0023)  time: 0.7139  data: 0.0022  max mem: 4143
    Epoch: [24]  [400/529]  eta: 0:01:32  lr: 0.001000  loss: 0.0874 (0.0899)  loss_classifier: 0.0276 (0.0315)  loss_box_reg: 0.0576 (0.0558)  loss_objectness: 0.0001 (0.0004)  loss_rpn_box_reg: 0.0022 (0.0023)  time: 0.7134  data: 0.0022  max mem: 4143
    Removed invalid box tensor([ 79, 264,  79, 267]) of 4 from image index 1018
    Epoch: [24]  [500/529]  eta: 0:00:20  lr: 0.001000  loss: 0.0888 (0.0911)  loss_classifier: 0.0321 (0.0316)  loss_box_reg: 0.0542 (0.0564)  loss_objectness: 0.0002 (0.0004)  loss_rpn_box_reg: 0.0020 (0.0028)  time: 0.7136  data: 0.0021  max mem: 4143
    Epoch: [24]  [528/529]  eta: 0:00:00  lr: 0.001000  loss: 0.0878 (0.0912)  loss_classifier: 0.0285 (0.0316)  loss_box_reg: 0.0554 (0.0565)  loss_objectness: 0.0001 (0.0004)  loss_rpn_box_reg: 0.0023 (0.0028)  time: 0.6984  data: 0.0021  max mem: 4143
    Epoch: [24] Total time: 0:06:17 (0.7137 s / it)
    creating index...
    index created!
    Test:  [  0/113]  eta: 0:01:26  model_time: 0.3522 (0.3522)  evaluator_time: 0.0063 (0.0063)  time: 0.7660  data: 0.3069  max mem: 4143
    Test:  [100/113]  eta: 0:00:04  model_time: 0.2940 (0.2946)  evaluator_time: 0.0071 (0.0075)  time: 0.3053  data: 0.0020  max mem: 4143
    Test:  [112/113]  eta: 0:00:00  model_time: 0.2942 (0.2946)  evaluator_time: 0.0075 (0.0075)  time: 0.3056  data: 0.0019  max mem: 4143
    Test: Total time: 0:00:35 (0.3102 s / it)
    Averaged stats: model_time: 0.2942 (0.2946)  evaluator_time: 0.0075 (0.0075)
    Accumulating evaluation results...
    DONE (t=0.13s).
    IoU metric: bbox
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.653
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.917
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.807
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.417
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.636
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.536
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.131
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.643
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.730
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.417
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.708
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.667
    SAVING PLOTS COMPLETE...
    SAVING PLOTS COMPLETE...
    SAVING PLOTS COMPLETE...
    SAVING PLOTS COMPLETE...
    SAVING PLOTS COMPLETE...
    SAVING PLOTS COMPLETE...


# 4. Results and Evaluation


```python
# import required libraries
import matplotlib.pyplot as plt
import glob as glob
%matplotlib inline
```


```python
# plot FasterRCNN output images
results_dir_path = '/root/ObjectDetection-FasterRCNN/outputs/training/plantdetection'
valid_images = glob.glob(f"{results_dir_path}/*.jpg")

for i in range(2):
    plt.figure(figsize=(10, 7))
    image = plt.imread(valid_images[i])
    plt.imshow(image)
    plt.axis('off')
    plt.show()
```


    
![png](/assests/faster_rcnn_lettuce/output_21_0.png)
    



    
![png](/assests/faster_rcnn_lettuce/output_21_1.png)
    



```python
# Model evaluation
!python eval.py --weights outputs/training/plantdetection/best_model.pth --config data_configs/custom_data.yaml --model fasterrcnn_resnet50_fpn_v2 --verbose
```

    100%|███████████████████████████████████████████| 29/29 [00:35<00:00,  1.23s/it]
    
    
    {'classes': tensor([1, 2, 3, 4, 5], dtype=torch.int32),
     'map': tensor(0.6547),
     'map_50': tensor(0.9272),
     'map_75': tensor(0.8034),
     'map_large': tensor(0.6149),
     'map_medium': tensor(0.6408),
     'map_per_class': tensor([0.4806, 0.6794, 0.7296, 0.7144, 0.6696]),
     'map_small': tensor(0.3659),
     'mar_1': tensor(0.1298),
     'mar_10': tensor(0.6450),
     'mar_100': tensor(0.7368),
     'mar_100_per_class': tensor([0.6189, 0.7392, 0.8029, 0.7856, 0.7377]),
     'mar_large': tensor(0.6673),
     'mar_medium': tensor(0.7146),
     'mar_small': tensor(0.4028)}
    
    
    ("Classes: ['__background__', 'Ready', 'empty_pod', 'germination', 'pod', "
     "'young']")
    
    
    AP / AR per class
    -------------------------------------------------------------------------
    |    | Class                | AP                  | AR                  |
    -------------------------------------------------------------------------
    |1   | Ready                | 0.481               | 0.619               |
    |2   | empty_pod            | 0.679               | 0.739               |
    |3   | germination          | 0.730               | 0.803               |
    |4   | pod                  | 0.714               | 0.786               |
    |5   | young                | 0.670               | 0.738               |
    -------------------------------------------------------------------------
    |Avg                        | 0.655               | 0.737               |


## Inference on validation and test samples


```python
!python inference.py --input data/pascal/valid/100011.jpg --weights outputs/training/plantdetection/best_model.pth
```

    Building from model name arguments...
    Test instances: 1
    Image 1 done...
    --------------------------------------------------
    TEST PREDICTIONS COMPLETE
    



```python
image = plt.imread('outputs/inference/res_1/100011.jpg')
plt.imshow(image)
plt.axis('off')
plt.show()
```


    
![png](/assests/faster_rcnn_lettuce/output_25_0.png)
    



```python
!python inference.py --input data/pascal/valid/100028.jpg --weights outputs/training/plantdetection/best_model.pth
```

    Building from model name arguments...
    Test instances: 1
    Image 1 done...
    --------------------------------------------------
    TEST PREDICTIONS COMPLETE
    



```python
image = plt.imread('outputs/inference/res_2/100028.jpg')
plt.imshow(image)
plt.axis('off')
plt.show()
```


    
![png](/assests/faster_rcnn_lettuce/output_27_0.png)
    



```python
!python inference.py --input data/pascal/test/100001.jpg --weights outputs/training/plantdetection/best_model.pth
```

    Building from model name arguments...
    Test instances: 1
    Image 1 done...
    --------------------------------------------------
    TEST PREDICTIONS COMPLETE
    



```python
image = plt.imread('outputs/inference/res_3/100001.jpg')
plt.imshow(image)
plt.axis('off')
plt.show()
```


    
![png](/assests/faster_rcnn_lettuce/output_29_0.png)
    



```python
!python inference.py --input data/pascal/test/100006.jpg --weights outputs/training/plantdetection/best_model.pth
```

    Building from model name arguments...
    Test instances: 1
    Image 1 done...
    --------------------------------------------------
    TEST PREDICTIONS COMPLETE
    



```python
image = plt.imread('outputs/inference/res_4/100006.jpg')
plt.imshow(image)
plt.axis('off')
plt.show()
```


    
![png](/assests/faster_rcnn_lettuce/output_31_0.png)
    


# End of the Notebook
