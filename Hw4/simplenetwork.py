import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

#Define hyperparemeters
batch_size = 128
learning_rate = 0.01
num_epochs = 20
input_size = 784  #28x28 pixels
hidden_size1 = 300
hidden_size2 = 200
num_classes = 10

class simplenetwork:
    def __init__(self):
        #define the model
        #The torch.randn function generates a tensor of size (hidden_size1, input_size) 
        #filled with random values drawn from a normal distribution with mean 0 and standard deviation 1
        #The division by np.sqrt(input_size) scales the initial weights by the square root of the input size, 
        #which is a common practice to ensure that the initial weights are not too large or too small.
        
        #Load the MNIST dataset 
        #TorchVision also offers a lot of handy transformations, such as cropping or normalization.
        self.train_dataset = datasets.MNIST(root= './data', train= True, transform= transforms.ToTensor(), download= True)
        self.test_dataset = datasets.MNIST(root = './data', train = False, transform= transforms.ToTensor())

        #Create data loaders
        self.train_loader = DataLoader(dataset= self.train_dataset, batch_size= batch_size, shuffle= True)
        self.test_loader = DataLoader(dataset= self.test_dataset, batch_size= batch_size, shuffle=False)
        
        self.W1 = torch.randn(hidden_size1, input_size) / np.sqrt(input_size)
        self.W2 = torch.randn(hidden_size2, hidden_size1) / np.sqrt(hidden_size1)
        self.W3 = torch.randn(num_classes, hidden_size2) / np.sqrt(hidden_size2)
        self.a1 = None
        self.a2 = None

    def forward(self, x):
        z1 = torch.matmul(self.W1, x)
        self.a1 = torch.sigmoid(z1)
        z2 = torch.matmul(self.W2, self.a1)
        self.a2 = torch.sigmoid(z2)
        z3 = torch.matmul(self.W3, self.a2)
        y_hat = torch.softmax(z3)
        return y_hat
    
    def train(self):
        #Train the model
        for epoch in range(num_epochs):
            # labels is a tensor of shape (batch_size,) that contains the true labels for a batch of images.
            for i, (images, labels) in enumerate(self.train_loader):
                #flatten the images:
                images = images.reshape(-1, input_size).T
                labels = labels.numpy() # converting the labels tensor from a PyTorch tensor object to a numpy array.

                #convert labels to one-hot vectors
                y = np.zeros((num_classes, batch_size))
                y[labels, np.arange(batch_size)] = 1
                x, y = torch.tensor(images), torch.tensor(y)

                #forward pass 
                y_hat = self.forward(x)

                #compute the loss
                loss = -torch.sum(y * torch.log(y_hat))

                #backward pass
                delta3 = y_hat - y
                delta2 = torch.matmul(self.W3.T, delta3) * self.a2 * (1- self.a2)
                delta1 = torch.matmul(self.W2.T, delta2) * self.a1 * (1-self.a1)
                dW3 = torch.matmul(delta3, self.a2.T) 
                dW2 = torch.matmul(delta2, self.a1.T)
                dW1 = torch.matmul(delta1, x.T)

                #update weights
                self.W1 -= learning_rate* dW1
                self.W2 -= learning_rate* dW2
                self.W3 -= learning_rate * dW3

                #print the loss every 100 iterations        