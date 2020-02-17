#!/usr/bin/python
# -*- coding: utf-8 -*-
# DevPSU Project 1
# Artificial Neural Network
# NeuralNetwork.py

# Import numpy library as np here

import numpy as np


# Functions used for Propagation

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return (x * (1.0 - x))


# Here is the logic of the Neural Network

class NeuralNetwork:

    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):

        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1

        d_weights2 = np.dot(self.layer1.T, 2 * (self.y - self.output)
                            * sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y
                            - self.output)
                            * sigmoid_derivative(self.output),
                            self.weights2.T)
                            * sigmoid_derivative(self.layer1))

        # update the weights with the derivative (slope) of the loss function

        self.weights1 += d_weights1
        self.weights2 += d_weights2


# Test Cases

#if __name__ == '__main__':
    #X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    #y = np.array([[0], [0], [0], [1]])

    # y = np.array([[0], [1], [1], [1]]) # Using the network for an OR gate
    # y = np.array([[1], [1], [1], [0]]) # Using the network for an NAND gate

    #nn = NeuralNetwork(X, y)

    #for i in range(1000):
    # for i in range(10000): # Increasing the number of trials
        #nn.feedforward()
        #nn.backprop()

    #print(nn.output)

# Custome State Machine
if __name__ == '__main__':
    # Taking training data
    inputs = []
    output = []
    print('Enter data in the format: A B X')
    for i in range(4):
        inp = input().split(' ')
        A = int(inp[0])
        B = int(inp[1])
        out = int(inp[2])
        inputs.append([A,B])
        output.append([out])

    inputs = np.array(inputs)
    output = np.array(output)

    nn = NeuralNetwork(inputs, output)

    for i in range(1000):
        nn.feedforward()
        nn.backprop()

    print(nn.output)

# Extra Challenge
# 1. If we increase the number of trials from 1000, the printed output gets closer to the actual output
# 2. i) Change the output to 0, 1, 1, 1
#   ii) Change the output to 1, 1, 1, 0
# 3. Add custom machine state based on user's input


