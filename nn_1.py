import numpy as np # helps with the math
import matplotlib.pyplot as plt # to plot error during training

# input data

import pandas as pd

dataset = pd.read_excel(io="trainx.xlsx", sheet_name='Sheet1')

x1i1 = dataset.iloc[2:20,0].values
x2i1 = dataset.iloc[2:20,1].values
x1i2 = dataset.iloc[2:20,2].values
x2i2 = dataset.iloc[2:20,3].values

inputs = []
for i in range(len(x1i1)):
    inputs.append([x1i1[i], x2i1[i]])

for i in range(len(x1i1)):
    inputs.append([x1i2[i], x2i2[i]])

#print(len(inputs))
inputs = np.array(inputs)

outputs = []

for i in range(len(x1i1)):
    outputs.append([0])

for i in range(len(x1i1)):
    outputs.append([1])

print(len(outputs))
outputs = np.array(outputs)

print(inputs)
print(outputs)

# create NeuralNetwork class
class NeuralNetwork:

    # intialize variables in class
    def __init__(self, inputs, outputs):
        self.inputs  = inputs
        self.outputs = outputs
        # initialize weights as .50 for simplicity
        self.weights = np.array([[.50], [.50]])
        self.error_history = []
        self.epoch_list = []

    #activation function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # data will flow through the neural network.
    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    # going backwards through the network to update weights
    def backpropagation(self):
        self.error  = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, deriv=True)
        self.weights += np.dot(self.inputs.T, delta)

    # train the neural net for 25,000 iterations
    def train(self, epochs=2500):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    # function to predict output on new and unseen input data
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction

# create neural network
NN = NeuralNetwork(inputs, outputs)
# train neural network
NN.train()



# create two new examples to predict                                   
example = np.array([[0.4574, 0.6667]])
example_2 = np.array([[0.8085, 1]])
example_3 = np.array([[1, 0.2652]])

# print the predictions for both examples                                   
print(NN.predict(example), ' - Correct: ', example[0][0])
print(NN.predict(example_2), ' - Correct: ', example_2[0][0])
print(NN.predict(example_3), ' - Correct: ', example_3[0][0])

# plot the error over the entire training duration
#plt.figure(figsize=(15,5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()

