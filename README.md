# Chest X-Ray Diagnosis

<p align="center">
  <img src="https://github.com/OmarReda/Chest-X-Ray-Diagnosis/blob/main/image.jpg" width="500">
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
  <img src="https://github.com/OmarReda/Chest-X-Ray-Diagnosis/blob/main/Diagram.png" width="700">
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
  <img src="https://github.com/OmarReda/Chest-X-Ray-Diagnosis/blob/main/source.gif"><img width="250" src="https://github.com/OmarReda/Chest-X-Ray-Diagnosis/blob/main/covid-19.gif">
</p>


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
x_train, x_test, y_train, y_test = train_test_split(Dataset,Y , test_size=0.3, random_state=40)
```

## Encodeing & Normalization
```python
trainY = to_categorical(trainY, 3)
testY = to_categorical(testY, 3)
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0
```

## Essential Comparisons

| Softmax     | Sigmoid     |
| :----------: | :----------: |
| Used for multi-classification in logistic regression model.	| Used for binary classification in logistic regression model. |
| The probabilities sum will be 1  | The probabilities sum need not be 1. |
| Used in the different layers of neural networks.	| Used as activation function while building neural networks. |
| The high value will have the higher probability than other values.	|  The high value will have the high probability but not the higher probability. |

| CategoricalCrossentropy class | BinaryCrossentropy class |
| :---------------------------: | :----------------------: |
| Computes the crossentropy loss between the labels and predictions. | Computes the cross-entropy loss between true labels and predicted labels. |
| Use this crossentropy loss function when there are two or more label classes. | Use this cross-entropy loss when there are only two label classes (assumed to be 0 and 1).|
| ``` tf.keras.losses.categorical_crossentropy( y_true, y_pred, from_logits=False, label_smoothing=0 ) ``` | ``` tf.keras.losses.binary_crossentropy( y_true, y_pred, from_logits=False, label_smoothing=0 ) ``` |

## Testing

```python
def testing (modelx,testX,testY):       
  (loss, acc) =  modelx.evaluate(testX, testY)
  print("[INFO] test accuracy: {:.4f}".format(acc))
```

## Base Models Initializations
```python
# Fewer Layers (CNN)
cnn_model_fewlayer=  models.Sequential()
# Many Layers (CNN)
cnn_model_manylayer=  models.Sequential()

# Fewer Layers (FCN)
few_layer_model = models.Sequential()
# Many Layers (FCN)
many_layer_model = models.Sequential()
```

## Improved Models Initializations (Pre-Trained)
```python
# Inception V3 
base_model = InceptionV3(input_shape = (256, 256, 3), include_top = False, weights = 'imagenet')
# ResNet50
base_model = ResNet50(include_top=False, weights='imagenet')  
# VGG16
base_model = VGG16(input_shape = (256, 256, 3), include_top = False, weights = 'imagenet')
```

## Results

| Models          | Layers         | Loss           | Test Accuracy  | F1-Score  | Recall   | Precision  |
| :-------------- | :------------- | :-------------:| :-------------:| :-------: | :------: | :--------: |
| Fully Connected | Few Layers     | 00%            |  00%           |  00%      | 00%      |  00%       |
| Fully Connected | Many Layers    | 00%            |  00%           |  00%      | 00%      |  00%       |
| CNN             | Few Layers     | 00%            |  00%           |  00%      | 00%      |  00%       |
| CNN             | Many Layers    | 00%            |  00%           |  00%      | 00%      |  00%       |
|                                                                                                        |
| ResNet18        | Pre-Trained    |  00%           |  00%           |  00%      | 00%      |  00%       |
| ResNet50        | Pre-Trained    |  00%           |  00%           |  00%      | 00%      |  00%       |
| Inception v3    | Pre-Trained    |  00%           |  00%           |  00%      | 00%      |  00%       |
| VGG16           | Pre-Trained    |  00%           |  00%           |  00%      | 00%      |  00%       |

## Confusion Matrix

* #TODO

<hr>

## Contributors

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
 
 <p align="center"><img width="150" src="https://github.com/OmarReda/Chest-X-Ray-Diagnosis/blob/main/tenor.gif"></p>



