# Chest X-Ray Diagnosis

<p align="center">
  <img src="https://github.com/ahmedsamy1234/Chest-X-Ray-Diagnosis/blob/main/image.jpg" width="500">
</p>

## Problem Statement
Given the chest X-ray dataset, our goal is to build a range of neural networks to diagnose chest X-rays. Our database consists of patients suffering from COVID-19, Pneumonia, and normal Patients (3 classes).

## Basic Implemented Models
* Fully Connected Neural Network Models.
  - Many Layers
  - Few Layers
* Convolutional Neural Network Models With A Classification Layer At The End.
  - Many Layers
  - Few Layers

## Improved Models
* Famous backbones.
  - ResNet 18-50
  - Inception V3
  - VGG 16

<p align="center">
  <img src="https://github.com/ahmedsamy1234/Chest-X-Ray-Diagnosis/blob/main/Diagram.png" width="700">
</p>

## Dataset
* Find the used dataset<a href="https://www.kaggle.com/tawsifurrahman/covid19-radiography-database"> here</a>

## Libraries 
* TensorFlow
* Keras
* Numpy
* Sklearn
* PIL
* Matplotlib

<p align="center">
  <img src="https://github.com/ahmedsamy1234/Chest-X-Ray-Diagnosis/blob/main/source.gif"><img width="250" src="https://github.com/ahmedsamy1234/Chest-X-Ray-Diagnosis/blob/main/covid-19.gif">
</p>


## Notebook Structure
```
├── Loading Images
│   ├── API File
│   └── Extract Images
├── Imports
├── Pathes
├── Initialization & Images Reading
├── Augmentation
│   ├── Covid Augmentation
│   ├── Normal Augmentation
│   └── Viral Augmentation
├── Constructing Labels & Dataset Arrays
├── Splitting and Categorization
│   ├── Split
│   ├── Categorization (Labels' Encodeing )
│   └── Normalization
├── Training loop
│   ├── HyperParameters
│   ├── Step Function
│   └── Main Training Function
├── Testing
│   └── Reporting Model Performance
├── Basic Models
│   ├── CNN Few Layers Model
|   |   ├── Layers
│   │   ├── With Augmentation
│   │   |   ├── Test Accuracy
│   │   |   └── Report
|   |   ├── With Augmentation
│   │   |   ├── Test Accuracy
│   │   |   └── Report
│   ├── CNN Many Layers Model
|   |   ├── Layers
│   │   ├── With Augmentation
│   │   |   ├── Test Accuracy
│   │   |   └── Report
|   |   ├── With Augmentation
│   │   |   ├── Test Accuracy
│   │   |   └── Report
│   ├── FCN Many Layers Model
|   |   ├── Layers
│   │   ├── With Augmentation
│   │   |   ├── Test Accuracy
│   │   |   └── Report
|   |   ├── With Augmentation
│   │   |   ├── Test Accuracy
│   │   |   └── Report
│   └── FCN Few Layers Model
|   |   ├── Layers
│   │   ├── With Augmentation
│   │   |   ├── Test Accuracy
│   │   |   └── Report
|   |   ├── With Augmentation
│   │   |   ├── Test Accuracy
│   │   |   └── Report
└── Pre-Trained Models
    ├── VGG 16
    │   ├── Unweighted
    |   |   └── Test Acurracy  
    │   ├── Weighted (Freezing Ealier Layers)
    |   |   └── Test Acurracy  
    |   └── Weighted  All Layers  Is Traniable
    |   |   └── Test Acurracy
    ├── ResNet 50
    │   ├── Unweighted
    |   |   └── Test Acurracy  
    │   ├── Weighted 
    |   |   └── Test Acurracy  
    └── Inception V3
        ├── Freezing
        └── Test Acurracy
```


## Equations
```python
Precision = True Positive / True Positive + False Positive
Recall (or Sensitivity) = True Positive / True Positive + False Negative
F1 Score = 2 ∗ (Precision ∗ Recall / Precision + Recall)
Accuracy = True positive + True Negative / True positive + False Negative + True Negative + False Positive
```

## Loss Function
```python
L(Y,Y^) = −(∑Y∗log (Y^) + (1−Y) ∗ loglog(1−Y^))
Where, Y = true label, Y^ = Predicted Labels, and L(Y,Y^) is loss function.
```

## Data Augmentation
Data Augmentation is a technique to increase the diversity of your training set by applying random (but realistic) transformations such as image rotation.
```python
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"), 
  layers.experimental.preprocessing.RandomRotation(0.2),
]) 
```

## Constructing Labels & Dataset Arrays
```python
Dataset = np.concatenate((COVID,Normal,Viral))
Y = np.concatenate((Covidlabel,Normallabel,Virallabel)) 
```

## Splitting Data
```python
x_train, x_test, y_train, y_test = train_test_split(Dataset,Y , test_size=0.3,random_state=40, stratify=Y)
x_train, x_val, y_train, y_val= train_test_split(x_train,y_train, test_size=0.2, random_state=1,stratify=y_train)

```

## Encodeing & Normalization
```python
trainY = to_categorical(trainY, 3)
testY = to_categorical(testY, 3)
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0
```

## Essential Comparisons

| Problem Type     | Output Type                        | Final Activation Function | Loss Function          |
| :--------------  | :--------------------------------  | :-----------------------  | :--------------------  |
| Regression 	     | Numerical Value                    | Linear	                  | Mean Squar Error (MSE) |
| Classsification  | Binary Outcome                     | Sigmoid                   |  Binary Cross Entropy  |
| Classsification  | Single Label, Multiple Classes	    | Softmax                   |  Cross Entropy         |
| Classsification  | Multiple Labels, Multiple Classes	| Sigmoid                   |  Binary Cross Entropy  |

| Softmax                                                             | Sigmoid                                                                        |
| :------------------------------------------------------------------ | :----------------------------------------------------------------------------- |
| Used for multi-classification in logistic regression model.	        | Used for binary classification in logistic regression model.                   |
| The probabilities sum will be 1                                     | The probabilities sum need not be 1.                                           |
| Used in the different layers of neural networks.	                  | Used as activation function while building neural networks.                    |
| The high value will have the higher probability than other values.	|  The high value will have the high probability but not the higher probability. |

| CategoricalCrossentropy class | BinaryCrossentropy class |
| :---------------------------  | :----------------------  |
| Computes the crossentropy loss between the labels and predictions. | Computes the cross-entropy loss between true labels and predicted labels. |
| Use this crossentropy loss function when there are two or more label classes. | Use this cross-entropy loss when there are only two label classes (assumed to be 0 and 1).|
| ``` tf.keras.losses.categorical_crossentropy( y_true, y_pred, from_logits=False, label_smoothing=0 ) ``` | ``` tf.keras.losses.binary_crossentropy( y_true, y_pred, from_logits=False, label_smoothing=0 ) ``` |


## Graphs
<img width="500" src="https://github.com/ahmedsamy1234/Chest-X-Ray-Diagnosis/blob/main/graphs.png">


<h2 align="center">Optimizers Types</h2>

### Gradient Descent 
* Gradient descent is an optimization algorithm that's used when training a machine learning model. It's based on a convex function and tweaks its parameters iteratively to minimize a given function to its local minimum.
### Momentum  
* Momentum is like a ball rolling downhill. The ball will gain momentum as it rolls down the hill. 
* Momentum helps accelerate Gradient Descent(GD) when we have surfaces that curve more steeply in one direction than in another direction.
### Nesterov accelerated gradient(NAG)
* Nesterov acceleration optimization is like a ball rolling down the hill but knows exactly when to slow down before the gradient of the hill increases again.
### Adagrad — Adaptive Gradient Algorithm
* Adagrad is an adaptive learning rate method. In Adagrad we adopt the learning rate to the parameters. We perform larger updates for infrequent parameters and smaller updates for frequent parameters.
* It is well suited when we have sparse data as in large scale neural networks. GloVe word embedding uses adagrad where infrequent words required a greater update and frequent words require smaller updates.
### Adadelta
* Adadelta is an extension of Adagrad and it also tries to reduce Adagrad’s aggressive, monotonically reducing the learning rate.
* It does this by restricting the window of the past accumulated gradient to some fixed size of w. Running average at time t then depends on the previous average and the current gradient.
### RMSProp 
* RMSProp is Root Mean Square Propagation. It was devised by Geoffrey Hinton. 
* RMSProp tries to resolve Adagrad’s radically diminishing learning rates by using a moving average of the squared gradient. It utilizes the magnitude of the recent gradient descents to normalize the gradient.  
### Adam — Adaptive Moment Estimation 
* Another method that calculates the individual adaptive learning rate for each parameter from estimates of first and second moments of the gradients.
* It also reduces the radically diminishing learning rates of Adagrad
* Adam can be viewed as a combination of Adagrad, which works well on sparse gradients and RMSprop which works well in online and nonstationary settings.
* Adam implements the exponential moving average of the gradients to scale the learning rate instead of a simple average as in Adagrad. It keeps an exponentially decaying average of past gradients
### Nadam- Nesterov-accelerated Adaptive Moment Estimation
* Nadam combines NAG and Adam
* Nadam is employed for noisy gradients or for gradients with high curvatures
* The learning process is accelerated by summing up the exponential decay of the moving averages for the previous and current gradient

<p align="center">
  <img src="https://cs231n.github.io/assets/nn3/opt1.gif" width="500">
</p>
  
  
## Hyperparameter 
```python
EPOCHS =  10                                                               
BS = 64                                                                         
INIT_LR = 1e-3
Optimizer = Adam
Loss_Function = CrossEntropy
# The Number Of Layers Differs From One Model To Another
```

## Improved Models Initializations (Pre-Trained)
```python
# Inception V3 
base_model = InceptionV3(input_shape = (256, 256, 3), include_top = False, weights = 'imagenet') first  200 layer is false(freezing)
  
# ResNet50
base_model = ResNet50(input_shape = (256, 256, 3),include_top=False, weights='imagenet')  
base_model = ResNet50(input_shape = (256, 256, 3),include_top=False, weights='None)  

# VGG16
base_model = VGG16(input_shape = (256, 256, 3), include_top = False, weights = 'imagenet')
base_model = VGG16(input_shape = (256, 256, 3), include_top = False, weights = 'imagenet') && all trainable layers is false (freezing)
base_model = VGG16(input_shape = (256, 256, 3), include_top = False, weights = None)  
```

## Training Loop 

```python
def trainingloop(model,trainX,trainY,numbrofbatches,EPOCHS,optimizer) :
  numUpdates = int(trainX.shape[0] / numbrofbatches) #1
  epoch_loss_avg=[]
  
  # loop over the number of epochs
  for epoch in range(0, EPOCHS):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    accuracy_for_each_epoah=np.array([])
    sys.stdout.flush()
  # loop over the data in batch size increments
  for i in range(0, numUpdates):
    # determine starting and ending slice indexes for the current
    # batch
    start = i * numbrofbatches
    end = start + numbrofbatches
  # take a step
  loss=step(model,trainX[start:end], trainY[start:end],optimizer)#5
  epoch_loss_avg.update_state(loss)
```

## Step function
```python 
def step(model,x, y,optimizer):
# keep track of our gradients
  with tf.GradientTape() as tape:
    pred = model(x)
    loss = categorical_crossentropy(y, pred)
  grads = tape.gradient(loss,model.trainable_variables)
  optimizer.apply_gradients(zip(grads,model.trainable_variables))
  return loss  
```  

## Testing
```python
def testing (modelx,testX,testY):       
  (loss, acc) =  modelx.evaluate(testX, testY)
  print("[INFO] test accuracy: {:.4f}".format(acc))
```

## Pre-Trained Models Strategies
1. **Train the entire model**, you use the architecture of the pre-trained model and train it. you’ll need a large dataset (and a lot of computational power).
2. **Train some layers and leave the others frozen**, as you remember, lower layers refer to general features, while higher layers refer to specific features. Here, we play with that dichotomy by choosing how much we want to adjust the weights of the network (a frozen layer does not change during training).  
  *  *Small data set -> leave more layers frozen to avoid overfitting.*					
  *  *Large dataset -> improve your model by training more layers.*
3. **Fixed feature extraction mechanism**, the main idea is to keep the convolutional base in its original form and then use its outputs to feed the classifier. 


## Results

### Basic Models Results
 
| Models | Layers | Augmentation | Loss | Test Accuracy | F1-Score Class0 | Recall Class0 | Precision Class0 | F1-Score Class1 | Recall Class1 | Precision Class1 | F1-Score Class2 | Recall Class2 | Precision Class2 |
| :-------------- | :--------- | :----:| :-------:| :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | 
| Fully Connected | Few Layers | Yes   |  1.0535  |  77.66%  |  79.18%  |  88.75%  |  71.48%  |  84.04%  |  95.83%  |  74.84%  |  62.78%  |  46.39%  |  97.09%  |
| Fully Connected | Many Layers| Yes   |  0.4197  |  84.15%  |  82.02%  |  80.14%  |  83.99%  |  90.60%  |  97.78%  |  84.41%  |  79.32%  |  74.86%  |  84.35%  |
| CNN             | Few Layers | Yes   |  0.8994  |  85.75%  |  85.69%  |  85.69%  |  85.69%  |  89.19%  |  83.61%  |  95.56%  |  83.92%  |  89.17%  |  79.26%  |
| CNN             | Many Layers| Yes   |  0.3850  |  90.18%  |  89.35%  |  81.53%  |  94.07%  |  90.66%  |  97.78%  |  84.51%  |  87.07%  |  86.67%  |  88.76%  |
| Fully Connected | Few Layers | No    |  2.3815  |  82.76%  |  80.32%  |  98.51%  |  67.81%  |  91.48%  |  85.00%  |  90.03%  |  74.15%  |  62.13%  |  91.94%  |
| Fully Connected | Many Layers| No    |  0.7163  |  88.78%  |  88.03%  |  82.34%  |  94.87%  |  92.51%  |  87.50%  |  98.13%  |  85.87%  |  95.54%  |  77.98%  |
| CNN             | Few Layers | No    |  0.2063  |  94.28%  |  93.43%  |  97.26%  |  89.89%  |  96.68%  |  96.94%  |  96.41%  |  92.37%  |  88.37%  |  96.75%  |
| CNN             | Many Layers| No    |  0.2039  |  93.33%  |  94.33%  |  95.27%  |  93.41%  |  93.53%  |  88.33%  |  99.38%  |  91.90%  |  95.54%  |  88.53%  |

### Improved Models Results

| Models | Layers | Loss | Test Accuracy | F1-Score Class0 | Recall Class0 | Precision Class0 | F1-Score Class1 | Recall Class1 | Precision Class1 | F1-Score Class2 | Recall Class2 | Precision Class2 |
| :---------- | :------------------------- | :----:| :----:| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| ResNet 50   | Pre-Trained Weighted       |  0.2736  |  97.03%  |  96.86%  |  96.02%  |  97.72%  |  98.74%  |  98.33%  |  99.16%  |  96.58%  |  97.77%  |  95.41%  |
| ResNet 50   | Pre-Trained Unweighted     |  0.2559  |  94.08%  |  93.28%  |  91.54%  |  95.09%  |  97.02%  |  96.11%  |  98.03%  |  92.18%  |  94.80%  |  98.70%  |
| VGG 16      | Pre-Trained Weighted       |  0.1652  |  95.05%  |  95.10%  |  96.52%  |  93.72%  |  85.89%  |  93.89%  |  97.97%  |  92.97%  |  93.32%  |  92.63%  |
| VGG 16      | Pre-Trained Unweighted     |  0.2316  |  92.38%  |  92.29%  |  98.26%  |  87.00%  |  94.22%  |  90.56%  |  98.19%  |  91.07%  |  88.37%  |  93.95%  |
| VGG 16      | Pre-Trained Freezing Layers|  0.2088  |  95.06%  |  93.58%  |  99.75%  |  88.13%  |  98.34%  |  98.89%  |  97.80%  |  91.88%  |  95.40%  |  99.42%  |
| Inception V3| Pre-Trained Freezing Layers|  0.1634  |  96.40%  |  94.48%  |  91.53%  |  97.63%  |  98.95%  |  98.61%  |  99.30%  |  94.77%  |  98.06%  |  91.69%  |

<hr>

<!--
## Contributors
-->
<!--
<table>
  <tr>
    <td align="center">
    <a href="https://github.com/ahmedsamy1234" target="_black">
    <img src="https://avatars.githubusercontent.com/ahmedsamy1234" width="150px;" alt="ahmed samy"/>
    <br />
    <sub><b>Ahmed Samy</b></sub></a><br />
    </td>
    <td align="center">
    <a href="https://github.com/OmarReda" target="_black">
    <img src="https://avatars.githubusercontent.com/OmarReda" width="150px;" alt="omar reda"/>
    <br />
    <sub><b>Omar Reda</b></sub></a><br />
    </td>
  </tr>
 </table>
-->

<p align="center">
  <img width="150" src="https://github.com/ahmedsamy1234/Chest-X-Ray-Diagnosis/blob/main/tenor.gif">
</p>



