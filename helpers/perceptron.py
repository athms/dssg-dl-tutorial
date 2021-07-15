#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm


def step(x):
    """step function: 
    outputs 1 if x > 0
    and -1 otherwise"""
    xout = np.ones_like(x)
    xout[x<0] = -1
    return xout


def sigmoid(x):
    """sigmoid function"""
    return 1.0/(1.0 + np.exp(-x)) 


class cross_entropy_loss:
    def __init__(self):
        self.name = 'cross-entropy'
        
    def loss(self, y, y_pred, zerotol=1e-10):
        """the cross-entropy loss for each 
        data instance
        
        Args:
            y (array): labels for each instance (0 or 1)
            y_pred (array): predicted probabilty that
                each instance belongs to class 1
        """
        loss = -(y * np.log(y_pred + zerotol) + (1 - y) * np.log(1 - y_pred + zerotol))
        return loss
    
    def derivative_loss(self, y, y_pred):
        """the derivative of the cross-entropy loss w.r.t. 
        to sigmoid activation function (we will get to 
        this later)
        
        Args:
            y (array): labels for each instance (0 or 1)
            y_pred (array): predicted probabilty that
                each instance belongs to class 1
        """
        return y_pred - y


class Perceptron:
    
    def __init__(self, n_in, activation=sigmoid, loss=cross_entropy_loss, b=None):
        """A simple Perceptron implementation;
        This implementation also contains the
        gradient descent algorithm (see the 
        gradient_descent_step and train 
        functions).
        
        Args:
            n_in (int): number of input features
            activation (function): activation function of the Perceptron;
                should only take x as input
            loss (function): the used loss function; 
                This should be the cross_entropy 
                loss for a sigmoid activation
            b (float): bias term; if a value is specified, the
                bias term is fixed at this value. if not, 
                the bias will be estimated during training.
        """
        self.n_in = n_in
        self.w = np.random.uniform(-1,1,n_in)
        if b is None:
            self.b = np.random.uniform(-1,1,1)
            self.fit_b = True
        else:
            self.b = b
            self.fit_b = False
        self.activation = activation
        self.loss = loss().loss
        self.derivative_loss = loss().derivative_loss

    def predict(self, x):
        """Predict probability that each 
        instance of x (with shape n_instances x n_features)
        belongs to class 1
        
        Args:
            x (ndarray): input data (n_instances x n_features)
            
        Returns:
            predicted probability for each instance
        """
        self.Z = np.dot(x, self.w) + self.b
        self.A = self.activation(self.Z)
        return self.A
    
    def gradient_descent_step(self, x, y, learning_rate):
        """A single gradient descent step.
        
        Args:
            x (ndarray): input data (n_instances x n_features)
            y (array): label of each instance (0 or 1)
            learning_rate (float): learning rate of the
                gradient descent algorithm
        """
        # compute derivative of loss wrt Z
        dZ = self.derivative_loss(y, self.predict(x))
        dW = np.dot(dZ, x)
        # subtract average derivative from weights
        self.w -= learning_rate * 1.0/dW.shape[0] * dW
        if self.fit_b:
            self.b -= learning_rate * (1.0/x.shape[0] * np.sum(dZ))
            
    def train(self, X, y, batch_size=8, learning_rate=1, n_steps=100):
        """Apply gradient descent algorithm.
        At each iteration, the algorithm will draw 
        a random sample from x and perform a weight 
        update with the partial derivatives that
        are computed by the use of this sample.
        
        Args:
            x (ndarray): input data (n_instances x n_features)
            y (array): label of each instance (indicated by 0 and 1)
            learning_rate (float): learning rate of the 
                gradient descent algorithm
            n_steps (int): number of gradient descent 
                iterations to perform during training
        """
        self.training_w = np.zeros((n_steps, self.n_in+1))
        self.training_loss = np.zeros(n_steps)
        for s in tqdm(range(n_steps)):
            # draw a random batch
            batch_idx = np.random.choice(X.shape[0], batch_size, replace=False)
            # compute and store mean loss
            self.training_loss[s] = np.mean(self.loss(y[batch_idx], self.predict(X[batch_idx])))
            # store current weights
            self.training_w[s,:self.n_in] = self.w
            self.training_w[s,-1] = self.b
            # perform a gradient descent step
            self.gradient_descent_step(X[batch_idx], y[batch_idx], learning_rate)