# Handwritten Mathematical Symbol Classification using SVM and ANN

## 1. Introduction

This report compares the results of the Support Vector Machine (SVM) and Artificial Neural Network (ANN) model on a handwritten mathematical symbol classification task. Each model was trained and evaluated on three different features: the Histogram of Oriented Gradients (HoG), Local Binary Patterns (LBP) and Raw pixels. These features have unique patterns, and the HoG feature performs best in this task. Regarding the model, SVM shows robust results on HoG and Raw Pixels, whereas ANN performs well on the LBP feature.

## 2. Dataset

### 2.1. Data exploration

The data set contains ten folders or classes, and each folder has 500 images; in total, 5000 images were assigned. The image size of each image is 45x45 with 3 channels (45,45,3). The illustration of sample images is shown in the Figure 1 below. 

![Imgur](https://imgur.com/TzQQFZU.png)
*Figure 1. Sample images of each class*


### 2.2. Data preparation steps

Data preparation includes label creation from subfolders, label encoding, train and test set split, and feature extraction for given images.

* Label creation. To train the model, create the labels from subfolders using the folder name. 

* Label encoding. Using Scikit-Learn’s Label Encoder method, the categorical labels were encoded into numeric representations. The encoded dictionary is as follows: 
`{'!': 0, '(': 1, '+': 2, 'beta': 3, 'cos': 4, 'log': 5, 'pi': 6, 'pm': 7, 'tan': 8, 'theta': 9}`

* Train and test split. After preparing the labels and dataset, split it into train and test sets with a test size of 30% of the original dataset. After splitting the original dataset, the training dataset has 3500 samples, and the test dataset has 1500 samples. 

### 2.3. Feature extraction

* HoG parameters are configured as below:
Orientation = 9, pixels_per_cell = (10,10), cells_per_block = (2,2) <br>
* LBP parameters are configured as below: 
`NumPoint = 32`, `Radius = 6`

For the LBP feature, experimented with different parameter settings. NumPoint = 32, Radius = 8 gives 0.738 accuracy score, NumPoint = 24, Radius = 8 gives accuracy score of 0.7127, NumPoint = 8, Radius = 3 gives accuracy of 0.6520 on SVM. The best configuration is NumPoints = 32 and Radius = 6, which gives an accuracy of 0.746.

## 3. Experimental results and discussion

### 3.1. Experimental settings: 

SVM parameters

The GridSearchCV method is used to select the best parameters in SVM. Parameter grid dictionary: `{'C': [1, 50, 100], 'kernel': [ 'linear', 'poly', 'rbf']}`.

The selected parameters are shown in Figure 2. 


|| HOG	| LBP	| Raw Input | 
---| ---- | ----|---|
|C	|1	| 100	| 1
|Kernel |	poly |	poly |	poly

*Figure 2. The parameters of SVM*

In this dataset, the “Polynomial” kernel is suitable instead of linear or radial basis function. 

ANN architectures and parameters

 ||   HOG	|LBP	| Raw Input|
 |--|----|---|---|
Input shapes|	3500, 324|	3500, 34|	3500, 2025
Number of neurons	|128, 128|	512, 512, 512|	128, 128
Number of hidden layers|	2	|3	|2|
Activation functions	|ReLu	|ReLu	|ReLu|
Output layers|	10, Softmax|	10, Softmax|	10, Softmax|

*Figure 3. Architecture of ANN*

The same optimizer, loss function, metrics and callbacks are configured for each   
feature. 
* optimizer = ‘adam’
* loss=‘sparse_categorical_crossentropy’
* metrics = ‘accuracy’
* callbacks (loss <0.1)
The number of epochs configured is 10 for HoG, 100 for LBP, and 50 for Raw Pixels. 


![Imgur](https://imgur.com/TYXfxqr.png)
*Figure 4. Visualisations of ANN model architecture*

### 3.2. Experimental results

#### 3.2.1. Confusion matrix
The confusion matrix reports the table between the true and predicted labels. The combinations of features and classifiers, along with their test accuracy, are shown in the figure below. 

![Imgur](https://imgur.com/CM5h6rB.png)

*Figure 5. Confusion matrixes of SVM and ANN on HoG, LBP, and Raw Pixel features.*

For ANN, it is mostly confused with predicting other classes as “!”, whereas SVM better distinguish the “!” from others. 

True classified examples of SVM are shown in Figure 6. 

![Imgur](https://imgur.com/jnz1IfT.png)
*Figure 6. True classified examples of SVM on HoG feature.*


#### 3.2.2. Comparative study 

Each model gives different results on training and test datasets. The comparisons of the accuracy of the SVM and ANN are shown in the tables below. 

Train set accuracy

|Classifier/Feature|	HOG	|LBP	|Raw Input|
|---|---|----|----|
SVM	|0.9980	|0.7537	| 1.0000
ANN	|0.9831	|0.8671	| 0.9725

*Figure 7. Train set accuracy of SVM and ANN.*

Test set accuracy

|Classifier/Feature	|HOG	|LBP	|Raw Input|
|--|---|---|---|
SVM	| 0.9833	| 0.7460 |	0.9813 |
ANN |	0.9607 |	0.7893 |	0.9113 |

*Figure 8. Test set accuracy of SVM and ANN.*

In this dataset, the best-performing feature is HoG, followed by Raw input and LBP. SVM showed the best result in HoG and Raw input features, while ANN outperformed SVM in the LBP feature. 


#### 3.2.3. Discussion

The results show notable variations in performance across different feature-classifier combinations.
Firstly, the HoG feature extraction method demonstrated superior performance compared to the LBP approach. This result aligns with expectations, considering that HoG is adept at capturing edge and gradient information, which are crucial for distinguishing between handwritten symbols. Conversely, LBP, which emphasises texture patterns, may be less effective when dealing with images dominated by edges rather than textures. In addition, the raw pixel intensities also yielded competitive results, ranking as the second-best performer in this experiment. 
Furthermore, SVM outperformed ANN in the HoG and raw pixel feature spaces with accuracy scores of 0.9833 and 0.9813. SVMs are renowned for their capability to find optimal decision boundaries in high-dimensional feature spaces, making them particularly well-suited for this task. While ANNs offer the potential to capture complex nonlinear relationships between features and class labels, SVMs with a Polynomial kernel showcased superior performance in this specific experiment, possibly due to the dataset’s relatively small size or the feature spaces’ intrinsic characteristics.

Misclassifications / Confusion Analysis:

Several instances of misclassifications or confusion were observed during the experimentation and evaluation. 

“pm” vs “+”. This misclassification suggests that the algorithm may struggle to differentiate between symbols with similar visual characteristics, particularly when they appear in contexts with limited contextual patterns.

“!” vs “(“. The misclassification between “!” and “(“ underscores the challenges associated with distinguishing between symbols with subtle visual differences, especially when they possess similar structural components.

“Theta” vs “+”. It highlights the difficulty in discriminating between symbols that share visual similarities, particularly when handwritten and subject to variations in writing styles. 

The examples of misclassified images are shown in Figure 9 and Figure 10. 

![Imgur](https://imgur.com/xXH2PTJ.png)

*Figure 9. Misclassified examples of SVM on HoG feature*


![Imgur](https://imgur.com/5Q4PWP5.png)
*Figure 10. Misclassified examples of ANN on LBP feature*

## Conclusion 

In conclusion, this study on handwritten mathematical symbol classification using HoG, LBP, and raw pixel intensities, combined with SVM and ANN, has provided valuable insights into the effectiveness of different feature extraction methods and classification algorithms. The HoG features outperformed LBP and raw pixel representations, demonstrating their efficacy in capturing edge and gradient information crucial for symbol discrimination. Also, raw pixel intensities yielded competitive results, highlighting the importance of simple and comprehensive feature representations. Furthermore, SVM exhibited superior performance compared to ANN in HoG and raw pixel feature spaces. Overall, this experiment can contribute to developing robust and accurate handwritten symbol recognition systems with potential applications in OCR, educational technology, and document processing.

### The code implementation can be found here: [Link](https://zagarsuren.github.io/2024/03/23/math-symbol-hog-code.html)