* [What is deep learning?](#a)
* [What are the main differences between AI, Machine Learning, and Deep Learning?](#b)
* [Differentiate supervised and unsupervised deep learning procedures.](#c)
* [What do you mean by "overfitting"?](#d)
* [What is Backpropagation?](#e)
* [Explain Data Normalization](#f)
* [How many layers in the neural network?](#g)
* [What is the use of the Activation function?](#h)
* [How many types of activation function are available?](#i)
* [What is the sigmoid function?](#j)
* [What is ReLU function?](#k)
* [What is the use of leaky ReLU function?](#l)
* [What is the softmax function?](#m)
* [What is the most used activation function?](#n)
* [What do you mean by Dropout?](#o)
* [Explain the following variant of Gradient Descent: Stochastic, Batch, and Mini-batch?](#p)
* [What do you understand by a convolutional neural network?](#q)

## What is deep learning?  <a name="a"></br>
Deep learning is a part of machine learning with an algorithm inspired by the structure and function of the brain, which is called an artificial neural network.

## What are the main differences between AI, Machine Learning, and Deep Learning?  <a name="b"></br>
AI stands for Artificial Intelligence. It is a technique which enables machines to mimic human behavior.
Machine Learning is a subset of AI which uses statistical methods to enable machines to improve with experiences.
Deep learning is a part of Machine learning, which makes the computation of multi-layer neural networks feasible. It takes advantage of neural networks to simulate human-like decision making.

## Differentiate supervised and unsupervised deep learning procedures.  <a name="c"></br>
Supervised learning is a system in which both input and desired output data are provided. Input and output data are labeled to provide a learning basis for future data processing.
Unsupervised procedure does not need labeling information explicitly, and the operations can be carried out without the same. The common unsupervised learning method is cluster analysis. It is used for exploratory data analysis to find hidden patterns or grouping in data.

## What do you mean by "overfitting"?  <a name="d"></br>
Overfitting is the most common issue which occurs in deep learning. It usually occurs when a deep learning algorithm apprehends the sound of specific data. It also appears when the particular algorithm is well suitable for the data and shows up when the algorithm or model represents high variance and low bias.

## What is Backpropagation?  <a name="e"></br>
Backpropagation is a training algorithm which is used for multilayer neural networks. It transfers the error information from the end of the network to all the weights inside the network. It allows the efficient computation of the gradient.


It can forward propagation of training data through the network to generate output.
It uses target value and output value to compute error derivative concerning output activations.
It can backpropagate to compute the derivative of the error concerning output activations in the previous layer and continue for all hidden layers.
It uses the previously calculated derivatives for output and all hidden layers to calculate the error derivative concerning weights.
It updates the weights.

## Explain Data Normalization.  <a name="f"></br>
Data normalization is an essential preprocessing step, which is used to rescale values to fit in a specific range. It assures better convergence during backpropagation. In general, data normalization boils down to subtracting the mean of each data point and dividing by its standard deviation.

## How many layers in the neural network?    <a name="g"></br>
Input Layer
The input layer contains input neurons which send information to the hidden layer.
Hidden Layer
The hidden layer is used to send data to the output layer.
Output Layer
The data is made available at the output layer.

## What is the use of the Activation function?  <a name="h"></br>
The activation function is used to introduce nonlinearity into the neural network so that it can learn more complex function. Without the Activation function, the neural network would be only able to learn function, which is a linear combination of its input data.
Activation function translates the inputs into outputs. The activation function is responsible for deciding whether a neuron should be activated or not. It makes the decision by calculating the weighted sum and further adding bias with it. The basic purpose of the activation function is to introduce non-linearity into the output of a neuron.

## How many types of activation function are available?  <a name="i"></br>
Binary Step
Sigmoid
Tanh
ReLU
Leaky ReLU
Softmax
Swish

## What is the sigmoid function?  <a name="j"></br>
The sigmoid activation function is also called the logistic function. It is traditionally a trendy activation function for neural networks. The input data to the function is transformed into a value between 0.0 and 1.0. Input values that are much larger than 1.0 are transformed to the value 1.0. Similarly, values that are much smaller than 0.0 are transformed into 0.0. The shape of the function for all possible inputs is an S-shape from zero up through 0.5 to 1.0. It was the default activation used on neural networks, 

## What is ReLU function?  <a name="k"></br>
A node or unit which implements the activation function is referred to as a rectified linear activation unit or ReLU for short. Generally, networks that use the rectifier function for the hidden layers are referred to as rectified networks.
Adoption of ReLU may easily be considered one of the few milestones in the deep learning revolution.

## What is the use of leaky ReLU function?  <a name="l"></br>
The Leaky ReLU (LReLU or LReL) manages the function to allow small negative values when the input is less than zero.

## What is the softmax function?  <a name="m"></br>
The softmax function is used to calculate the probability distribution of the event over 'n' different events. One of the main advantages of using softmax is the output probabilities range. The range will be between 0 to 1, and the sum of all the probabilities will be equal to one. When the softmax function is used for multi-classification model, it returns the probabilities of each class, and the target class will have a high probability.

## What is the most used activation function?  <a name="n"></br>
Relu function is the most used activation function. It helps us to solve vanishing gradient problems.

## What do you mean by Dropout?  <a name="o"></br>
Dropout is a cheap regulation technique used for reducing overfitting in neural networks. We randomly drop out a set of nodes at each training step. As a result, we create a different model for each training case, and all of these models share weights. It's a form of model averaging.

## Explain the following variant of Gradient Descent: Stochastic, Batch, and Mini-batch?   <a name="p"></br>
#### Stochastic Gradient Descent
Stochastic gradient descent is used to calculate the gradient and update the parameters by using only a single training example.
#### Batch Gradient Descent
Batch gradient descent is used to calculate the gradients for the whole dataset and perform just one update at each iteration.
#### Mini-batch Gradient Descent
Mini-batch gradient descent is a variation of stochastic gradient descent. Instead of a single training example, mini-batch of samples is used. Mini-batch gradient descent is one of the most popular optimization algorithms.

## What do you understand by a convolutional neural network?  <a name="q"></br>
A convolutional neural network, often called CNN, is a feedforward neural network. It uses convolution in at least one of its layers. The convolutional layer contains a set of filter (kernels). This filter is sliding across the entire input image, computing the dot product between the weights of the filter and the input image. As a result of training, the network automatically learns filters that can detect specific features.

## Explain the different layers of CNN.
There are four layered concepts that we should understand in CNN (Convolutional Neural Network):
#### Convolution
This layer comprises of a set of independent filters. All these filters are initialized randomly. These filters then become our parameters which will be learned by the network subsequently.
#### ReLU
The ReLu layer is used with the convolutional layer.
#### Pooling
It reduces the spatial size of the representation to lower the number of parameters and computation in the network. This layer operates on each feature map independently.
#### Full Collectedness
Neurons in a completely connected layer have complete connections to all activations in the previous layer, as seen in regular Neural Networks. Their activations can be easily computed with a matrix multiplication followed by a bias offset.











































































