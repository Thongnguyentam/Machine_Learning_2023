import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

class CustomeLogisticRegression:
    def __init__(self, learning_rate = 0.01):
        self.weights = None
        self.bias = None
        self.train_accuracies = []
        self.losses = []
        self.learning_rate = learning_rate


    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1/(1 + z)
        else:
            z = np.exp(x)
            return z/(1+z)
    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(z) for z in x])

    def loss_func(self, y_true, y_pred):
        #compute cross-entry
        y_zero_lost = y_true*np.log(y_pred + 1e-9) 
        y_one_lost = (1 - y_true)*np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_lost + y_one_lost)

    def gradient(self, x, y_true, y_pred):
        #take derivatives of binary cross entropy
        # (σ(z) − y)x
        diff = y_pred - y_true
        bias = np.mean(diff)
        weight = np.matmul(x.transpose(),diff)
        #weight has the shape of (input x output, number of input features and number of output classes)
        # Each row of this matrix corresponds to the gradient of the loss with respect to the weights that connect one input feature to all output classes.
        weight = np.array([np.mean(grad) for grad in weight]) # the mean of each row of gradients_w to obtain the gradient of the loss with respect to each weight.
        return weight, bias

    def update_model_param(self, error_w, error_b):
        self.weights = self.weights - self.learning_rate*error_w
        self.bias = self.bias - self.learning_rate*error_b

    def fit(self, x, y, epoch):
        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        for i in range(epoch):
            x_dot_weight = np.matmul(self.weights, x.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weight)
            loss = self.loss_func(y, pred)
            errorw, errorb = self.gradient(x, y, pred)
            self.update_model_param(errorw, errorb)
            #predict 1 if pred > 0.5 and 0 if otherwise
            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            
            self.train_accuracies.append(accuracy_score(y, pred_to_class))
            self.losses.append(loss)
    
    def predict(self, x):
        #self weights is a row vector that contains weight for each feature
        # x is a matrix where each row represents an example, and each column represents a feature.
        #x_dot_weight produces a row vector of predictions. Each element of the row vector corresponds to the prediction for a specific example.
        x_dot_weight = np.matmul(self.weights, x.transpose()) + self.bias
        probability = self._sigmoid(x_dot_weight)
        return [1 if p > 0.5 else 0 for p in probability]
    
    def predict_proba_lr(self, x):
        x_dot_weight = np.matmul(self.weights, x.transpose()) + self.bias
        positive_probability = self._sigmoid(x_dot_weight)
        # stack vertically
        # the first array contains the probability of the negative class
        # and the second array contains the probability of the positive class.
        return np.vstack([1-positive_probability, positive_probability]).transpose()
