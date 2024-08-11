# Mathematical Symbol Classification using HoG feature

## Outline
1. Environment preparation
2. Load Data
3. Histogram-of-Oriented Gradient (HOG) feature extraction
4. Support Vector Machine (SVM) model training and evaluation
5. Artificial Neural Network (ANN) model training and evaluation
6. References

# 1. Environment preparation


```python
# Import Numpy, Pandas, and Matplotlib for numeric computation, data processing and plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Scikit Learn
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Image processing and feature extraction
from skimage import feature
import cv2

# Directory and random variable generation
import os
import random
import csv

# Set for recproducibility
seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

%matplotlib inline
```


```python
from google.colab import drive
drive.mount('/content/gdrive')
```

    Mounted at /content/gdrive


# 2. Load Data


```python
# Define training and test dataset paths
y_train_path = 'y_train_category.npy'
y_test_path = 'y_test_category.npy'
X_train_path = 'X_train.npy'
X_test_path = 'X_test.npy'
```


```python
# Define DataLoader function
def DataLoader(y_train_path, y_test_path, X_train_path, X_test_path):
  '''
  This function reads the datasets from Google Drive and performs Label Encoding on y_train and y_test.
  '''
  # Load y_train and y_test
  y_train_category = np.load(y_train_path, allow_pickle=True)
  y_test_category = np.load(y_test_path, allow_pickle=True)

  # Define label encoder and fit
  le = LabelEncoder()
  le.fit(y_train_category)

  # Transform categories
  y_train = le.transform(y_train_category)
  y_test = le.transform(y_test_category)

  # create mapping of target classes
  mapping = dict(zip(le.classes_, range(len(le.classes_))))
  print(f"Mapping of Y: {mapping}")

  # Load X_train and X_test
  X_train = np.load(X_train_path)
  X_test = np.load(X_test_path)

  # Pring train and test data shape
  print(f"Training data shape: {X_train.shape}")
  print(f"Testing data shape: {X_test.shape}")

  return y_train, y_test, X_train, X_test
```


```python
# Read dataset
y_train, y_test, X_train, X_test = DataLoader(y_train_path, y_test_path, X_train_path, X_test_path)
```

    Mapping of Y: {'!': 0, '(': 1, '+': 2, 'beta': 3, 'cos': 4, 'log': 5, 'pi': 6, 'pm': 7, 'tan': 8, 'theta': 9}
    Training data shape: (3500, 45, 45)
    Testing data shape: (1500, 45, 45)


# 3. Histogram-of-Oriented Gradient (HoG) feature extraction


```python
# The Function to compute the HoG feature
def compute_hog_features(images):
    # create an empty list
    hog_features = []
    # loop through images
    for image in images:
        # Compute HoG features using skimage
        features = feature.hog(image, orientations =9, pixels_per_cell=(10,10), cells_per_block=(2,2),
                               transform_sqrt=True, block_norm="L2-Hys")
        # append computed features to the empty list
        hog_features.append(features)
    # convert list to NumPy array
    return np.array(hog_features)
```


```python
# Apply function on X_train data
X_train_hog = compute_hog_features(X_train)
```


```python
# Apply function on X_test data
X_test_hog = compute_hog_features(X_test)
```


```python
# Print shape of the extracted dataset
print(f"Train data shape: {np.shape(X_train_hog)}")
print(f"Test data shape: {np.shape(X_test_hog)}")
```

    Train data shape: (3500, 324)
    Test data shape: (1500, 324)


# 4. Support Vector Machine (SVM) model training and evaluation

## Parameter tuning


```python
# import GridSearchCV from sklearn.model_selection for Parameter tuning
from sklearn.model_selection import GridSearchCV
```


```python
# Parameter tuning function
def parameter_tuning(classifier_name, parameters):
  classifier = GridSearchCV(classifier_name, parameters)
  classifier.fit(X_train_hog, y_train)
  return print(f"Best params: {classifier.best_params_} and score: {classifier.best_score_:.4f}")
```


```python
# Import SVC from sklearn.svm
from sklearn.svm import SVC
# Parameter dictionary
parameters_svc = {'C': [1, 50, 100], 'kernel':[ 'linear', 'poly', 'rbf']}
# Parameter tuning
parameter_tuning(SVC(), parameters_svc)
```

    Best params: {'C': 1, 'kernel': 'poly'} and score: 0.9837


## Model training


```python
# Fit X_train_hog and y_train with the best parameters
model_svm = SVC(C=1, kernel = 'poly', random_state=42)
model_svm.fit(X_train_hog, y_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SVC(C=1, kernel=&#x27;poly&#x27;, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">SVC</label><div class="sk-toggleable__content"><pre>SVC(C=1, kernel=&#x27;poly&#x27;, random_state=42)</pre></div></div></div></div></div>




```python
# Train set accuracy
print("Train set accuracy SVC: {:.4f}".format(model_svm.score(X_train_hog, y_train)))
```

    Train set accuracy SVC: 0.9980


## Model evaluation on test data


```python
# Predict the X_test_hog data
y_pred_svm = model_svm.predict(X_test_hog)
```


```python
# Test set accuracy
accuracy_svm =  metrics.accuracy_score(y_test, y_pred_svm)
print("Test set accuracy SVC: {:.4f}".format(accuracy_svm))
```

    Test set accuracy SVC: 0.9833



```python
# Create the labelNames list
labelNames = ['!', '(', '+', 'beta', 'cos', 'log', 'pi', 'pm', 'tan', 'theta']
```


```python
# Display the confusion matrix and import required package
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

fig, ax = plt.subplots(figsize = (5,4), dpi = 100)
cm = confusion_matrix(y_test, y_pred_svm)
cmp = ConfusionMatrixDisplay(cm, display_labels = labelNames)
cmp.plot(ax=ax)
all_sample_title = 'Accuracy Score SVM (HoG): {:.4f}'.format(accuracy_svm)
plt.title(all_sample_title, size = 12);
```


    
![png](src/dl_hog/output_25_0.png)
    



```python
# Display the evaluation report
print(metrics.classification_report(y_test, y_pred_svm))
```

                  precision    recall  f1-score   support
    
               0       0.96      0.99      0.98       150
               1       0.99      0.97      0.98       150
               2       0.94      0.97      0.95       150
               3       0.99      1.00      1.00       150
               4       1.00      0.99      1.00       150
               5       0.99      1.00      0.99       150
               6       1.00      0.99      0.99       150
               7       0.97      0.97      0.97       150
               8       0.99      0.99      0.99       150
               9       0.99      0.96      0.98       150
    
        accuracy                           0.98      1500
       macro avg       0.98      0.98      0.98      1500
    weighted avg       0.98      0.98      0.98      1500
    


## Visualize SVM results


```python
def display_results_svm(image_indices):
  num_samples = X_test_hog[image_indices].shape[0]
  selected_features = X_test_hog[image_indices]
  selected_labels = [y_test[i] for i in image_indices]

  # Predict selected features
  predictions_svm = model_svm.predict(np.array(selected_features))

  # Select original images
  selected_original_images = [X_test[i] for i in image_indices]

  # Plot original images and display Actual label and Predicted label
  plt.figure(figsize=(15, 8))
  for i in range(8):
      plt.subplot(2, 4, i + 1)
      plt.imshow(selected_original_images[i], cmap='gray')
      plt.title("Predicted: {} vs Actual: {}".format(labelNames[predictions_svm[i]], labelNames[selected_labels[i]]))
      plt.axis('off')
  plt.show()
```


```python
# Create indices
true_indices_svm = np.where(y_pred_svm == y_test)[0]
misclassified_indices_svm = np.where(y_pred_svm != y_test)[0]
```


```python
# Display true classified results
display_results_svm(true_indices_svm)
```


    
![png](src/dl_hog/output_30_0.png)
    



```python
# Display  misclassified results
display_results_svm(misclassified_indices_svm)
```


    
![png](src/dl_hog/output_31_0.png)
    


# 5. Artificial Neural Network (ANN) model training and evaluation


```python
# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
```

    2.15.0



```python
# Seed TensorFlow
seed = 42
tf.random.set_seed(seed)
```

## Model training


```python
# Define callback
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.1):
      print("Early stopping Triggered")
      self.model.stop_training = True
callbacks = myCallback()
```


```python
# Model architecture
model_ann = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(X_train_hog.shape[1],)),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
model_ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
History = model_ann.fit(X_train_hog, y_train, epochs=10, callbacks=[callbacks])
```

    Epoch 1/10
    110/110 [==============================] - 2s 4ms/step - loss: 1.1085 - accuracy: 0.7277
    Epoch 2/10
    110/110 [==============================] - 0s 4ms/step - loss: 0.2760 - accuracy: 0.9237
    Epoch 3/10
    110/110 [==============================] - 1s 5ms/step - loss: 0.1715 - accuracy: 0.9503
    Epoch 4/10
    110/110 [==============================] - 0s 3ms/step - loss: 0.1287 - accuracy: 0.9660
    Epoch 5/10
    110/110 [==============================] - 0s 3ms/step - loss: 0.1044 - accuracy: 0.9711
    Epoch 6/10
     93/110 [========================>.....] - ETA: 0s - loss: 0.0817 - accuracy: 0.9778Early stopping Triggered
    110/110 [==============================] - 0s 2ms/step - loss: 0.0820 - accuracy: 0.9771



```python
model_ann.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 128)               41600     
                                                                     
     dense_1 (Dense)             (None, 128)               16512     
                                                                     
     dense_2 (Dense)             (None, 10)                1290      
                                                                     
    =================================================================
    Total params: 59402 (232.04 KB)
    Trainable params: 59402 (232.04 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________



```python
# Visualize ANN Model
modelViz_file= 'model_ann_hog.png'
keras.utils.plot_model(model_ann, to_file=modelViz_file, show_shapes=True)
```




    
![png](src/dl_hog/output_39_0.png)
    




```python
## Plot the learning curves
import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(History.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

## Plot only the loss train loss
plt.plot(History.history['loss'])
plt.ylabel('cost')
plt.xlabel('Epochs')
plt.title("Cost/Loss Curve")
plt.show()
```


    
![png](src/dl_hog/output_40_0.png)
    



    
![png](src/dl_hog/output_40_1.png)
    



```python
# Evaluate the model's performance on the training dataset.
model_ann.evaluate(X_train_hog, y_train)
```

    110/110 [==============================] - 0s 2ms/step - loss: 0.0658 - accuracy: 0.9831





    [0.0658428966999054, 0.9831428527832031]



## ANN evaluation on the test dataset


```python
# Evaluate the model's performance on the test dataset.
model_ann.evaluate(X_test_hog, y_test)
```

    47/47 [==============================] - 0s 2ms/step - loss: 0.1169 - accuracy: 0.9667





    [0.11694876104593277, 0.9666666388511658]




```python
# Predict X_test_hog
predict_ann = model_ann.predict(X_test_hog)
```

    47/47 [==============================] - 0s 2ms/step



```python
# Convert probabilities to labels
y_pred_ann = ((predict_ann > 0.5)*1).argmax(axis=1)
y_pred_ann
```




    array([7, 0, 8, ..., 5, 8, 3])




```python
# Compute accuracy
accuracy_ann =  metrics.accuracy_score(y_test, y_pred_ann)
print("Test data accuracy ANN: {:.4f}".format(accuracy_ann))
```

    Test data accuracy ANN: 0.9607



```python
# Display the confusion matrix and import required package
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix

fig, ax = plt.subplots(figsize = (5,4), dpi = 100)
cm = confusion_matrix(y_test, y_pred_ann)
cmp = ConfusionMatrixDisplay(cm, display_labels = labelNames)
cmp.plot(ax=ax)
all_sample_title = 'Accuracy Score ANN (HoG): {:.4f}'.format(accuracy_ann)
plt.title(all_sample_title, size = 12);
```


    
![png](src/dl_hog/output_47_0.png)
    



```python
# Display the evaluation report
print(metrics.classification_report(y_test, y_pred_ann))
```

                  precision    recall  f1-score   support
    
               0       0.84      0.99      0.91       150
               1       0.99      0.95      0.97       150
               2       0.96      0.87      0.91       150
               3       0.99      0.97      0.98       150
               4       0.99      0.96      0.98       150
               5       0.99      0.99      0.99       150
               6       0.99      0.97      0.98       150
               7       0.92      0.96      0.94       150
               8       0.96      0.99      0.98       150
               9       1.00      0.95      0.97       150
    
        accuracy                           0.96      1500
       macro avg       0.96      0.96      0.96      1500
    weighted avg       0.96      0.96      0.96      1500
    


## Visualize ANN results


```python
def display_results_ann(image_indices):
  num_samples = X_test_hog[image_indices].shape[0]
  selected_features = X_test_hog[image_indices]
  selected_labels = [y_test[i] for i in image_indices]

  # predict the selected features
  predictions_ann = model_ann.predict(np.array(selected_features))
  # Get class names from predictions
  y_pred_ann_display = ((predictions_ann > 0.5)*1).argmax(axis=1)

  # Select original images
  selected_original_images = [X_test[i] for i in image_indices]

  # plot results
  plt.figure(figsize=(15, 8))
  for i in range(8):
      plt.subplot(2, 4, i + 1)
      plt.imshow(selected_original_images[i], cmap='gray')
      plt.title("Predicted: {} vs Actual: {}".format(labelNames[y_pred_ann_display[i]], labelNames[selected_labels[i]]))
      plt.axis('off')
  plt.show()
```


```python
# Create indices
true_indices_ann = np.where(y_pred_ann == y_test)[0]
misclassified_indices_ann = np.where(y_pred_ann != y_test)[0]
```


```python
# True classification results
display_results_ann(true_indices_ann)
```

    46/46 [==============================] - 0s 2ms/step



    
![png](src/dl_hog/output_52_1.png)
    



```python
# Misclassification results
display_results_ann(misclassified_indices_ann)
```

    2/2 [==============================] - 0s 7ms/step



    
![png](src/dl_hog/output_53_1.png)
    


# 6. References

[1] Chollet, F., & others. (2015). Keras. https://keras.io<br/>
[2] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. http://www.deeplearningbook.org<br/>
[3] Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2. (Publisher link). <br/>
[4] Kingma, D. P., & Ba, J. (2017). Adam: A Method for Stochastic Optimization (arXiv:1412.6980). arXiv. http://arxiv.org/abs/1412.6980<br/>
[5] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., & Thirion, B. (2011).  Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825--2830.<br/>
[6] Stéfan van der Walt, Johannes L. Schönberger, Juan Nunez-Iglesias, François Boulogne, Joshua D. Warner, Neil Yager, Emmanuelle Gouillart, Tony Yu and the scikit-image contributors. scikit-image: Image processing in Python. PeerJ 2:e453 (2014) https://doi.org/10.7717/peerj.453


# The end of the notebook
