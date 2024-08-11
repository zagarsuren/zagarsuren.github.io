# Image Classification using VGG16 (Code)
 <br/>

# Outline
1. Environment preparation
2. Train, validation, test data generator
3. Customized CNN architecture
4. Model training
5. Results and Evaluation

# 1. Environment preparation


```python
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
import keras.utils as image
from tensorflow.keras import layers
from tensorflow.keras import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set for recproducibility
import os
import random
seed = 25077001
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

%matplotlib inline
```



# 2. Train, validation, test data generator


```python
train_folder = "/ImageClassification/train"
valid_folder = "/ImageClassification/valid"
test_folder = "/ImageClassification/test"
```


```python
train_datagen = ImageDataGenerator(rescale = 1./255)
valid_datagen = ImageDataGenerator(rescale = 1./255)
test_dategen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_folder,
                                                     target_size=(224, 224),  
                                                     batch_size=30,
                                                     class_mode='categorical')
validation_generator = valid_datagen.flow_from_directory(valid_folder,
                                                   target_size=(224, 224),  
                                                   batch_size=30,
                                                   class_mode='categorical')
test_generator = test_dategen.flow_from_directory(test_folder,
                                                   target_size=(224, 224),  
                                                   batch_size=30,
                                                   shuffle=False,
                                                   class_mode='categorical')
```

    Found 2326 images belonging to 20 classes.
    Found 348 images belonging to 20 classes.
    Found 660 images belonging to 20 classes.



```python
class_names = train_generator.class_indices
print(class_names)
```

    {'ALBERTS TOWHEE': 0, 'ANDEAN GOOSE': 1, 'Afghan_hound': 2, 'Border_terrier': 3, 'CHUKAR PARTRIDGE': 4, 'COLLARED ARACARI': 5, 'CUBAN TROGON': 6, 'English_foxhound': 7, 'FASCIATED WREN': 8, 'FIERY MINIVET': 9, 'GOLDEN EAGLE': 10, 'LARK BUNTING': 11, 'Lhasa': 12, 'Old_English_sheepdog': 13, 'PYGMY KINGFISHER': 14, 'Samoyed': 15, 'affenpinscher': 16, 'kuvasz': 17, 'papillon': 18, 'standard_schnauzer': 19}


# 3. Customized CNN architecture


```python
# import required libraries
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras import models
from keras import layers
from keras import optimizers

# create a conv base
conv_base = VGG16(weights='imagenet',
                  include_top=False, 
                  input_shape=(224, 224, 3))

# Create a Sequential model
model = Sequential()

# Iterate over layers in VGG16 model up to block5_conv1 or remove the block5
for layer in conv_base.layers[:-4]:
    model.add(layer)
# Add dropout
model.add(Dropout(0.5))

# Fully connected layer
model.add(Flatten())

# Add Dense layer and dropout
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

# Add Dense layer and dropout
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(20, activation='softmax'))
```


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
                                                                     
     block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
                                                                     
     block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
                                                                     
     block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
                                                                     
     block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
                                                                     
     block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
                                                                     
     block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
                                                                     
     block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                     
     block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                     
     block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
                                                                     
     block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
                                                                     
     block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                     
     block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                     
     block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
                                                                     
     dropout (Dropout)           (None, 14, 14, 512)       0         
                                                                     
     flatten (Flatten)           (None, 100352)            0         
                                                                     
     dense (Dense)               (None, 512)               51380736  
                                                                     
     dropout_1 (Dropout)         (None, 512)               0         
                                                                     
     dense_1 (Dense)             (None, 64)                32832     
                                                                     
     dropout_2 (Dropout)         (None, 64)                0         
                                                                     
     dense_2 (Dense)             (None, 20)                1300      
                                                                     
    =================================================================
    Total params: 59,050,132
    Trainable params: 59,050,132
    Non-trainable params: 0
    _________________________________________________________________


# 4. Model training


```python
# Compile a model
model.compile(loss='categorical_crossentropy',
              optimizer = RMSprop(learning_rate=1e-4),
              metrics=['acc'])
```


```python
!mkdir /root/42028/AT2/data/Image_Classification/weights
```


```python
# Define check pointing
filepath='/Image_Classification/weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
```


```python
# Define callback
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.1):
      print("Early stopping Triggered")
      self.model.stop_training = True
early_stopping = myCallback()
```


```python
# Train a model
history = model.fit(
    train_generator,
    steps_per_epoch = 10, # iteration
    epochs = 100,
    verbose = 2,

    validation_data = validation_generator,
    validation_steps = 10,
    callbacks = [early_stopping, checkpoint]
    )
```

    Epoch 1/100
    10/10 - 36s - loss: 4.3368 - acc: 0.0433 - val_loss: 2.9896 - val_acc: 0.0433 - 36s/epoch - 4s/step
    Epoch 2/100
    10/10 - 8s - loss: 3.0941 - acc: 0.0333 - val_loss: 2.9832 - val_acc: 0.0900 - 8s/epoch - 774ms/step
    Epoch 3/100
    10/10 - 6s - loss: 3.0216 - acc: 0.0567 - val_loss: 2.9837 - val_acc: 0.0600 - 6s/epoch - 589ms/step ...
  
    Epoch 98/100
    10/10 - 6s - loss: 0.6052 - acc: 0.8100 - val_loss: 0.9296 - val_acc: 0.7433 - 6s/epoch - 585ms/step
    Epoch 99/100
    10/10 - 6s - loss: 0.5591 - acc: 0.8467 - val_loss: 1.0359 - val_acc: 0.6967 - 6s/epoch - 586ms/step
    Epoch 100/100
    10/10 - 6s - loss: 0.3533 - acc: 0.8733 - val_loss: 0.9483 - val_acc: 0.7567 - 6s/epoch - 584ms/step


# 5. Results and Evaluation


```python
# Plot training accuracy and loss
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy (Customized CNN)')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss (Customized CNN)')
plt.legend()

plt.show()
```


    
![png](/assests/image_classification_cnn/output_19_0.png)
    



    
![png](/assests/image_classification_cnn/output_19_1.png)
    


### Train, validation, test set accuracy


```python
print("Train set loss and accuracy:" , model.evaluate(train_generator))
print("Validation set loss and accuracy:" , model.evaluate(validation_generator))
print("Test set loss and accuracy:" , model.evaluate(test_generator))
```


    78/78 [==============================] - 8s 101ms/step - loss: 0.0662 - acc: 0.9888
    Train set loss and accuracy: [0.06621439754962921, 0.9888219833374023]

    12/12 [==============================] - 4s 329ms/step - loss: 0.8965 - acc: 0.7644
    Validation set loss and accuracy: [0.896451473236084, 0.7643678188323975]
     1/22 [>.............................] - ETA: 4s - loss: 0.4714 - acc: 0.8667

    22/22 [==============================] - 2s 103ms/step - loss: 0.8516 - acc: 0.7727
    Test set loss and accuracy: [0.8516203761100769, 0.7727272510528564]


### Confusion matrix display on test set


```python
test_predict = model.predict(test_generator)
predicted_labels = np.argmax(test_predict, axis=1)
```

    22/22 [==============================] - 2s 101ms/step



```python
true_labels = test_generator.classes
```


```python
# Import required libraries
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Compute accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Compute confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title(f'Confusion Matrix for Customized CNN (Test set accuracy: {accuracy:.2f})')
plt.show()
```


    
![png](/assests/image_classification_cnn/output_25_0.png)
    



```python
# Display the evaluation report on test set
from sklearn import metrics
print(metrics.classification_report(true_labels, predicted_labels, target_names=class_names))
```

                          precision    recall  f1-score   support
    
          ALBERTS TOWHEE       0.87      0.87      0.87        31
            ANDEAN GOOSE       1.00      0.89      0.94        27
            Afghan_hound       0.78      0.74      0.76        47
          Border_terrier       0.86      0.56      0.68        34
        CHUKAR PARTRIDGE       0.91      0.94      0.92        32
        COLLARED ARACARI       0.94      0.97      0.96        33
            CUBAN TROGON       1.00      0.89      0.94        28
        English_foxhound       0.69      0.87      0.77        31
          FASCIATED WREN       0.93      0.83      0.88        30
           FIERY MINIVET       0.83      0.83      0.83        30
            GOLDEN EAGLE       0.88      0.97      0.92        31
            LARK BUNTING       0.91      1.00      0.95        31
                   Lhasa       0.60      0.57      0.58        37
    Old_English_sheepdog       0.62      0.30      0.41        33
        PYGMY KINGFISHER       0.78      0.91      0.84        32
                 Samoyed       0.54      0.84      0.65        43
           affenpinscher       0.62      0.87      0.72        30
                  kuvasz       0.95      0.63      0.76        30
                papillon       0.65      0.77      0.71        39
      standard_schnauzer       0.47      0.29      0.36        31
    
                accuracy                           0.77       660
               macro avg       0.79      0.78      0.77       660
            weighted avg       0.78      0.77      0.77       660
    


### Inferences on sample images


```python
# Function to preprocess the image
def preprocess_image(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    # Resize the image to match the input size of the model
    img = cv2.resize(img, (224, 224))
    # Preprocess the image
    img = img.astype('float32') / 255.0  
    # Expand the dimensions to match the input shape of the model
    img = np.expand_dims(img, axis=0)
    return img

# Function to predict the class label
def predict_single_image(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    # Get predictions
    predictions = model.predict(img)
    # Get the predicted class label
    predicted_class = np.argmax(predictions)
    return predicted_class, predictions
```


```python
# Path to single image
image_path = '/ImageClassification/test/affenpinscher/143.jpg'

img = image.load_img(image_path, target_size=(224, 224, 3))
plt.imshow(img)

# Predict the class label for the single image
predicted_class, predictions = predict_single_image(image_path)

# Display the predicted class label
print(print(
    "Predicted class: {}, Confidence: {:.2f}"
    .format(list(class_names.keys())[list(class_names.values()).index(np.argmax(predictions))], np.max(predictions))
))
```

    1/1 [==============================] - 1s 719ms/step
    Predicted class: affenpinscher, Confidence: 0.62
    None



    
![png](/assests/image_classification_cnn/output_29_1.png)
    



```python
# Path to single image
image_path = '/ImageClassification/test/ANDEAN GOOSE/12.jpg'

img = image.load_img(image_path, target_size=(224, 224, 3))
plt.imshow(img)

# Predict the class label for the single image
predicted_class, predictions = predict_single_image(image_path)

# Display the predicted class label
print(print(
    "Predicted class: {}, Confidence: {:.2f}"
    .format(list(class_names.keys())[list(class_names.values()).index(np.argmax(predictions))], np.max(predictions))
))
```

    1/1 [==============================] - 0s 28ms/step
    Predicted class: ANDEAN GOOSE, Confidence: 1.00
    None



    
![png](/assests/image_classification_cnn/output_30_1.png)
    



```python
# Path to single image
image_path = '/ImageClassification/test/Old_English_sheepdog/105.jpg'

img = image.load_img(image_path, target_size=(224, 224, 3))
plt.imshow(img)

# Predict the class label for the single image
predicted_class, predictions = predict_single_image(image_path)

# Display the predicted class label
print(print(
    "Predicted class: {}, Confidence: {:.2f}"
    .format(list(class_names.keys())[list(class_names.values()).index(np.argmax(predictions))], np.max(predictions))
))
```

    1/1 [==============================] - 0s 30ms/step
    Predicted class: Samoyed, Confidence: 0.55
    None



    
![png](/assests/image_classification_cnn/output_31_1.png)
    



```python
# Path to single image
image_path = '/ImageClassification/test/standard_schnauzer/23.jpg'

img = image.load_img(image_path, target_size=(224, 224, 3))
plt.imshow(img)

# Predict the class label for the single image
predicted_class, predictions = predict_single_image(image_path)

# Display the predicted class label
print(print(
    "Predicted class: {}, Confidence: {:.2f}"
    .format(list(class_names.keys())[list(class_names.values()).index(np.argmax(predictions))], np.max(predictions))
))
```

    1/1 [==============================] - 0s 31ms/step
    Predicted class: English_foxhound, Confidence: 0.39
    None



    
![png](/assests/image_classification_cnn/output_32_1.png)
    



```python
# Path to single image
image_path = '/ImageClassification/test/English_foxhound/124.jpg'

img = image.load_img(image_path, target_size=(224, 224, 3))
plt.imshow(img)

# Predict the class label for the single image
predicted_class, predictions = predict_single_image(image_path)

# Display the predicted class label
print(print(
    "Predicted class: {}, Confidence: {:.2f}"
    .format(list(class_names.keys())[list(class_names.values()).index(np.argmax(predictions))], np.max(predictions))
))
```

    1/1 [==============================] - 0s 25ms/step
    Predicted class: English_foxhound, Confidence: 0.97
    None



    
![png](/assests/image_classification_cnn/output_33_1.png)
    


# End of the Notebook
