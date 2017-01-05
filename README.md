# Convolutional Neural-Network for MNIST

An implementation of convolutional neural-network (CNN) for MNIST with various techniques such as data augmentation, dropout, batchnormalization, etc. 

## Network architecture

CNN with 4 layers has following architecture.

+ input layer : 784 nodes (MNIST images size)
+ first convolution layer : 5x5x32
+ first max-pooling layer
+ second convolution layer : 5x5x64
+ second max-pooling layer
+ third fully-connected layer : 1024 nodes
+ output layer : 10 nodes (number of class for MNIST)

## Tools for improving CNN performance

The following techniques are employed to imporve performance of CNN.

### Train
#### 1. Data augmentation

The number of train-data is increased to 5 times by means of</br>
+ Random rotation : each image is rotated by random degree in ranging [-15°, +15°].
+ Random shift : each image is randomly shifted by a value ranging [-2pix, +2pix] at both axises.
+ Zero-centered normalization : a pixel value is subtracted by (PIXEL_DEPTH/2) and divided by PIXEL_DEPTH.

#### 2. Parameter initializers
+ Weight initializer : xaiver initializer
+ Bias initializer : constant (zero) initializer
  
#### 3. Batch normalization
All convolution/fully-connected layers use batch normalization.

#### 4. Dropout
The third fully-connected layer employes dropout technique.

#### 5. Exponentially decayed learning rate
A learning rate is decayed every after one-epoch.

### Test
#### 1. Ensemble prediction
Every model makes a prediction (votes) for each test instance and the final output prediction is the one that receives the highest number of votes.

## Usage

### Train
`python mnist_cnn_train.py`

Training logs are saved in "logs/train".
Trained model is saved as "model/model.ckpt".

### Test a single model
`python mnist_cnn_test.py --model-dir <model_directory> --batch-size <batch_size> --use-ensemble False`

+ `<model_directory>` is the location where a model to be testes is saved. Please do not specify filename of "model.ckpt".
+ `<batch_size>` is employed to reduce burden of memory of machine. The number of test data is 10,000 for MNIST. Different batch_size gives the same result, but requiring different memory size.

You may command like `python mnist_cnn_test.py --model-dir model/model01_99.61 --batch-size 5000 --use-ensemble False`.

### Test ensemble prediction
`python mnist_cnn_test.py --model-dir <model_directory> --batch-size <batch_size> --use-ensemble True`

+ `<model_directory>` is the location of root directory. The root directory contains the sub-directories containg each model.

You may command like `python mnist_cnn_test.py --model-dir model --batch-size 5000 --use-ensemble True`.

## Simulation results

CNN with the same hyper-parameters has been trained 30 times, and gives the following results.
+ A single model : **99.61%** of accuracy.</br>(the model is saved in "model/model01_99.61".)
+ Ensemble prediction : **99.72%** of accuracy.</br>(All 5 models under "model/" are used. I found the collection of 5 models by try and error.)

**99.72%** of accuracy is the **5th** rank according to [Here](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html).

## Acknowledgement

This implementation has been tested on Tensorflow r0.12.
