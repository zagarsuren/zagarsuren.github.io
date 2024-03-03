# Mini AI Project: Obesity Classification using Logistic Regression and Neural Networks

## Table of Contents	
**Executive Summary**

**1.	Introduction**

**2.	Solution**

&nbsp; &nbsp;&nbsp;  2.1. Methods/AI techniques used

&nbsp; &nbsp;&nbsp; 2.2. Implementation of AI techniques

&nbsp; &nbsp;&nbsp; 2.3. Comparisons of different AI techniques

**3. Conclusions and Recommendations**

**4. List of References**

---

## Executive Summary

This report comprehensively analyses multiclass classification techniques using logistic regression and feedforward neural networks (FNNs) in the context of obesity data. The study aims to compare the performance of these two models and derive insights for obesity classification. The dataset was adapted from the UCI machine learning repository and consists of 2111 instances and 17 attributes of food consumption and physical activities data collected from Mexico, Peru, and Colombia.
The obesity data under consideration poses a significant public health challenge. Accurate classification of obesity levels is crucial for healthcare interventions. This report evaluates the application of logistic regression and FNNs for this purpose.
Section 2 introduces the methods and techniques used in the report, implementation, and comparison of AI techniques. The methods section explains the Logistic Regression and FNN algorithms in the context of multinomial classification problems, including the formulas and key parameters. Scikit Learn and Keras libraries were used to implement the chosen algorithms. Implementation involves data exploration, pre-processing, and model training and testing processes. The categorical encoding, normalisation, and train test splits are performed in data pre-processing. Overall, three models were trained and tested for this classification problem. Firstly, trained the Logistic Regression model with a maximum iteration of 1000. Two models with different numbers of neurons were used to evaluate the different parameter settings in the FNNs. FNN Model 1 has three hidden layers; each layer consists of 100, 100, and 50 neurons. FNN Model 2 is defined with the same number of hidden layers with 512, 512, and 100 neurons.
The results of the AI techniques are summarised in the comparison section. The four classification metrics – Precision, Recall, F-1 Score and Accuracy, are used to evaluate the model performance. The Logistic Regression results ranged between 0.88 and 0.89 on the test set and 0.86 on the future set. FNNs with 100, 100, and 50 neurons returned the best results with ~0.96 precision, recall, F-1 score and accuracy on the test set and 0.94 on the future set. The wider network with 512, 512 and 100 neurons showed a result of ~0.89 on the test and ~0.86 on the future set.
In summary, the choice of machine learning technique and model architecture can significantly impact the accuracy of multiclass classification in the context of obesity data. The FNN with 100, 100, and 50 neurons emerged as the most promising approach, offering high precision, recall, F-1 score, and accuracy. However, proper parameter tuning and architecture selection are crucial if a large dataset exists.

## 1. Introduction

Obesity is a global health concern with far-reaching implications for individuals, healthcare systems, and society. Accurate classification of obesity levels is fundamental for tailoring effective interventions and strategies. In this report, we explore multiclass classification techniques in the context of obesity data, aiming to implement predictive models to estimate obesity levels in a given dataset.
Obesity is associated with an elevated risk of several significant types of cancer, such as post-menopausal breast cancer, colorectal cancer, endometrial cancer, kidney cancer, esophageal cancer, pancreatic cancer, liver cancer, and gallbladder cancer. Extra body fat increases the likelihood of cancer-related deaths by about 17% (Pati et al., 2023). As such, it is imperative to develop robust and accurate classification models that can help identify obesity levels in individuals. Applying artificial intelligence (AI) techniques offers promising avenues for enhancing the precision of such classification tasks.
This report delves into implementing and evaluating two distinct multiclass classification techniques: logistic regression and FNNs. The logistic regression is a well-established and interpretable linear model known for its efficiency and simplicity. The FNNs represent the power of deep learning in capturing complex, nonlinear relationships within data.

**Data Source**

The obesity dataset is adapted from the UCI Machine Learning Repository. The dataset comprises information for assessing the obesity rates among people in Mexico, Peru, and Colombia, focusing on their dietary patterns and physical well-being. It consists of 17 different characteristics and a total of 2111 data entries. Each data entry is associated with a category called "NObeyesdad" or Obesity Level, which facilitates the classification of individuals into various groups, including Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, and Obesity Type III (Palechor & Manotas, 2019). 

**Data description**


The descriptions of the data set are summarised in the table below. 

| Feature	| Description|
|-----------|-----------|
|Gender |	Gender (Male, Female) |
| Age	| Age|
Height	|Height in meter |
Weight |	Weight in kilogram |
Family_history	| Family history with overweight (yes, no) |
FAVC |	Frequent consumption of high caloric food (yes, no) |
FCVC	| Frequency of consumption of vegetables |
NCP	| Number of main meals |
CAEC	|Consumption of food between meals (Sometimes, Frequently, Always, No) |
SMOKE	| Whether smoking or not (yes, no) |
CH2O	| Consumption of water daily |
SCC	| Calories consumption monitoring (yes, no) |
FAF	| Physical activity frequency |
TUE	| Time using technology devices |
CALC |	Consumption of alcohol (Sometimes, Frequently, Always, No) |
MTRANS	| Transportation used (Public transportation, Automobile,  Walking, Motorbike, Bike) |
NObeyesdad |	Obesity type (Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III) |

*Figure 1. Description of the dataset (UCI Machine Learning Repository)*

Classification problem formulation
The classification problem aims to find the hypothesis function h of true function f(x) in a given obesity dataset.
y=f(x)  → h(x)

Target (y): 
NObeyesdad – Obesity types (7 classes)

Features (X): 
x_1, x_2  ,x_3  ,...   – Age, height, weight, …, MTRANS.

The illustration of the workflow is shown in Figure 2.

![img](https://i.imgur.com/ywIxnY4.png)
*Figure 2. The workflow of the classification task*

## 2. Solution

This section will discuss the methods and techniques used, the implementation of AI techniques, and the results and comparisons. We will employ the Logistic Regression and FNNs to classify the obesity types. After implementing the algorithms, compare the result using common classification metrics – Precision, Recall, F-1 Score, and Accuracy. 

### 2.1. Methods/AI techniques used
#### 2.1.1.	Logistic Regression
The logistic regression model, found in academic literature under various names such as logit regression, maximum-entropy classification, or the log-linear classifier, employs a logistic function to predict the probabilities of the class (Pedregosa et al., 2011).  The logistic function transforms real-valued input to an output number y between 0 and 1, interpreted as the probability that the input object belongs to the positive class, given its input features x_1,〖 x〗_2… x_n (Peng et al., 2002). 

An illustration of logistic regression in the context of binary classification is shown below. 

![img](https://i.imgur.com/Rybv6O5.png)
*Figure 3. Logistic Regression*

The logistic regression is defined below:
![img](https://i.imgur.com/127pIaB.png)

The binary problem extended to multiple classes used to implement the multinomial logistic regression (Pedregosa et al., 2011). When, y_i∈1,..,K – Label encoded target feature for observation i, K – Number of classes, the prediction of class probabilities is defined in the formula below.

![img](https://i.imgur.com/F7Wk6cY.png)

The loss function to be optimised:
![img](https://i.imgur.com/IVO9HwC.png)

This report will use Scikit Learn’s LogisticRegression() module from the linear model to implement the multinomial logistic regression. 

#### 2.1.2.	Feedforward Neural Networks (FNNs)
![img](https://i.imgur.com/x7Zvq0t.png)
*Figure 4. Feedforward Neural Network in classification task*

**Activation function**

Similar to the regression problem, we will use the ReLU activation function in this report. ReLU activation function is one of the popular non-linear activation functions used in neural networks – as it outputs any negative input values into zeros. (Goodfellow et al., 2016). The ReLU activation function is defined below.  

ReLU(x) = max(0,x)

**Cost function**

We will use the “Sparse categorical cross-entropy” loss. It computes the cross-entropy loss between the target and the predictions when there are more than two classes (Chollet & others, 2015). 

**Optimiser**

An "optimiser" helps to train the neural network model and compute the optimal weights of the network (Russell & Norvig, 2021). This report utilises the popular learning algorithm Adam (Adaptive moment optimisation) optimiser in this problem (Kingma & Ba, 2017).

**Softmax**

The softmax function is used in a multiclass classification problem, representing the probability distribution over K different classes (Goodfellow et al., 2016). The softmax function is defined below, where K is the number of classes and z is the input vector. 

![img](https://i.imgur.com/dm9tTXY.png)

### 2.2. Implementation of AI techniques 

This section will perform the data exploration, pre-processing, and model training and testing. 

#### 2.2.1. Data exploration

The data consists of 2111 instances and 17 variables or columns. The first five rows of the dataset are shown in Figure 5. 

![img](https://i.imgur.com/ekCDN71.png)
*Figure 5. The first five rows of the dataset*

Figure 6 compares the obesity type by weight and age. The first chart shows a significant association between weight and obesity type. On the other hand, there is little correlation between age and obesity. 

![img](https://i.imgur.com/4L33xDG.png)
*Figure 6. Box plot of obesity*

Figure 7 shows the cross-tabulation between obesity and family history. 98%, 99.7%, and 100% of the surveyed people with obesity type I, II and III said their families had obesity.  
![img](https://i.imgur.com/p7BJn56.png)
*Figure 7. Crosstab of Obesity and Family History*

The correlation coefficients between features are illustrated in the figure below. 
![img](https://i.imgur.com/k6rjuxp.png)
*Figure 8. Correlation matrix*

Positively correlated features with Obesity

• Weight: 0.39 

• CAEC (Consumption of food between meals): 0.33

• Family history with obesity: 0.31 

Negatively correlated features with Obesity

• CALC (Consumption of alcohol): -0.13

• NCP (Number of main meals): -0.093

• SCC (Calories consumption monitoring): - 0.051


#### 2.2.2. Data pre-processing

In data pre-processing, employed the Scikit Learn library and used the LabelEncoder, StandardScaler, and train_test_split methods. 

**Label encoding of categorical features**

Gender, family history, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS, Nobeyesdad are all categorical types of features. The model training requires numeric input, meaning all categorical features must be converted to numbers. To perform this task, employed the LabelEncoder method. The encoded results of the Target feature – Nobeyesdad: 
{Insufficient_Weight: 0, Normal_Weight: 1, Obesity_Type_I: 2, Obesity_Type_II: 3, Obesity_Type_III: 4,  Overweight_Level_I: 5, Overweight_Level_II: 6}

**Z-Score normalisation**

The Z-score normalisation is applied to each feature to scale the different features. Employed the StandardScaler().fit_transform method from the preprocessing module.

**Train, validation, and test set split** 

To split the dataset, use the train_test_split method. The test set is 20% of the original dataset and extracted 100 samples from the test set to evaluate the performance after training the model. The size of the three sets is shown below.

• Training set size: 1688

• Test set size: 323

• Future set size: 100


#### 2.2.3.	Logistic Regression implementation and results
To implement the Logistic Regression model in a multinomial setting, used the Scikit Learn library. For multiclass classification, the LogisticRegression module automatically fits on the multiclass labels. The defined model is LogisticRegression(max_iter = 1000).

Results of the Logistic Regression Model, including the estimated coefficients for class 4 (Obesity Type III) are summarised below. 

![img](https://i.imgur.com/d3aNyIo.png)
*Figure 9. Confusion Matrix of Logistic Regression*

Classification report of Logistic Regression on test set is summarised below. 

![img](https://i.imgur.com/DifQDs6.png)

#### 2.2.4.	Feedforward Neural Networks implementation and results

The Keras library (Chollet & others, 2015) is used to implement the two FNN models with different numbers of neurons to evaluate the different parameter settings. Model 1 has three hidden layers; each layer consists of 100, 100, and 50 neurons. Model 2 is defined with the same number of hidden layers with 512, 512, and 100 neurons. Each hidden layer used the ReLU activation function. Then, define the number of epochs as 20 and batch size as 10. When compiling the model, employed sparse categorical cross-entropy as a loss function and used an Adam optimiser. The model architectures are shown below. 

| Model 1 architecture | Model 2 architecture|
|--------------------|----------------------- |
| Input Layer   |           |
|•	input_dimension – N. features = 16 | •	input_dimension – N. features = 16|
|Hidden Layer 1:|             |
|•	Number of Neurons: 100 | •	Number of Neurons: 512|
|•	Activation Function: ReLU  |•	Activation Function: ReLU |
| Hidden Layer 2: |               |
|•	Number of Neurons: 100 | •	Number of Neurons: 512 |
| •	Activation Function: ReLU | •	Activation Function: ReLU  |
| Hidden Layer 3: |                 |
| •	Number of Neurons: 50 | •	Number of Neurons: 100 |
|•	Activation Function: ReLU | •	Activation Function: ReLU  |
| Output Layer: |                  |
|•	Number of Neurons: 7 | •	Number of Neurons: 7 |
|•	Activation Function: Softmax | •	Activation Function: Softmax 	|


**Results of the FNNs:**
![img](https://i.imgur.com/9pmsGmz.png)
*Figure 10. Confusion matrix of FNNs*

Classification report on test set (Model 1):
![img](https://i.imgur.com/1twSIBt.png)

Classification report on test set (Model 2):
![img](https://i.imgur.com/IiQDE9G.png)

### 2.3.	Comparisons of different AI techniques

The results of the AI techniques are summarised in the table below. Firstly, the Logistic Regression metrics ranged between 0.88 and 0.89 on the test set and 0.86 on the future set.  FNNs with 100, 100, and 50 neurons returned the best results with ~0.96 precision, recall, F-1 score and accuracy on the test set and 0.94 on the future set. The wider network with 512, 512 and 100 neurons showed a result of ~0.89 on the test and ~0.86 on the future set. 

![img](https://i.imgur.com/rxyqXod.png)
*Figure 11. Results of the Logistic Regression and FNNs*


**Logistic Regression**	

Pros:

•	Interpretability. Logistic regression is easy to interpret, making it useful when understanding the relationship between input features and the output class probabilities.

•	Efficiency. Logistic regression is computationally efficient, especially for smaller datasets. It requires fewer parameters to estimate, which can be an advantage when there is limited data.

•	Low risk of overfitting: Logistic regression is less prone to overfitting than neural networks and decision tree models. In this report, Logistic regression showed stable results among test and future data sets.	

Cons:

•	Linearity. Logistic regression assumes a linear relationship. Therefore, it may not perform well if the true relationship is highly nonlinear.

•	Limited expressive power. Logistic regression may not capture patterns in the data, particularly when dealing with complex multiclass problems.


**FNNs**

Pros:

•	Nonlinearity. FNNs can model highly nonlinear relationships between input features and class probabilities, making them suitable for complex multiclass problems.

•	Parameters. FNNs are many parameters that can be tuned. The results of this report showed that FNNs can capture the relationship well even in a small dataset when tuned with the proper parameters and architecture. 

•	Scalability. FNNs can handle large and complex datasets effectively and can be scaled up by adding more layers and neurons for improved performance.	


Cons:

•	Complexity. Neural networks are inherently more complex than logistic regression models. Training and tuning them may require more computational resources and data.

•	Explainability. Neural networks are challenging to interpret their decision-making processes compared to logistic regression.


## 3.	Conclusions and Recommendations

**Conclusion**

In conclusion, the multiclass classification analysis of obesity data yielded valuable insights into the performance of different AI techniques. Logistic Regression demonstrated respectable results, with precision and recall ranging between 0.88 and 0.89 on the test set and 0.86 on the future set. However, the most noteworthy findings emerged from the FNNs.
Among the FNNs architectures, the model with 100, 100, and 50 neurons delivered an outstanding performance, boasting approximately 0.96 precision, recall, F-1 score, and accuracy on the test set, with 0.94 precision, recall, F-1 score, and accuracy on the future set. This suggests that well-configured FNNs can be highly effective for classifying obesity data.
Conversely, the wider network with 512, 512, and 100 neurons did not match the performance of the narrower FNNs, achieving approximately 0.89 precision on the test set and about 0.86 on the future set. These results highlight the importance of selecting an appropriate neural network architecture for the task, as wider networks may not yield superior results. The choice between neural network architecture depends on the specific problem and dataset. 
In summary, the choice of machine learning technique and model architecture can significantly impact the accuracy of multiclass classification in the context of obesity data. The FNNs with 100, 100, and 50 neurons emerged as the most promising approach, offering high precision, recall, F-1 score, and accuracy. However, proper parameter tuning and architecture selection are crucial if a large dataset exists.

**Recommendations**

Based on the findings from the multiclass classification analysis of obesity data, several recommendations can be made.
To potentially enhance the performance of the FNNs model, it is advisable to conduct further hyperparameter tuning. Adjusting parameters such as batch size and configuring early stopping may help fine-tune the model's performance. 
Explore the possibilities of feature engineering to improve the quality and diversity of the input data. For instance, creating new features, such as body mass index using weight and height features, could be helpful. These techniques can help capture more nuanced patterns in the obesity data and may further enhance the FNN's performance.
Consider implementing ensemble methods to combine the predictions of multiple models. Ensembles can often provide more robust and accurate results by leveraging the strengths of different models. Combining the FNNs with Logistic Regression or other models may yield better results.
Ensure the model's performance is robust by conducting rigorous cross-validation and thorough evaluation. Assess how the model generalises to different datasets and employ techniques such as k-fold cross-validation to validate its stability and accuracy.

---

## 4. List of References

Chollet, F., & others. (2015). Keras [Computer software]. https://keras.io

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. http://www.deeplearningbook.org

Kingma, D. P., & Ba, J. (2017). Adam: A Method for Stochastic Optimization (arXiv:1412.6980). arXiv. http://arxiv.org/abs/1412.6980

Palechor, F. M., & Manotas, A. D. L. H. (2019). Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico. Data in Brief, 25, 104344. https://doi.org/10.1016/j.dib.2019.104344

Pati, S., Irfan, W., Jameel, A., Ahmed, S., & Shahid, R. K. (2023). Obesity and Cancer: A Current Overview of Epidemiology, Pathogenesis, Outcomes, and Management. Cancers, 15(2), 485. https://doi.org/10.3390/cancers15020485

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., & Thirion, B. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825--2830.

Peng, C.-Y. J., Lee, K. L., & Ingersoll, G. M. (2002). An Introduction to Logistic Regression Analysis and Reporting. The Journal of Educational Research, 96(1), 3–14. https://doi.org/10.1080/00220670209598786

Russell, S., & Norvig, P. (2021). Artificial intelligence: A modern approach, global edition. Pearson Education, Limited.
