#####################################################################################################################
#   CS 6375.003 - Assignment 3, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler

class NeuralNet:
    def __init__(self, train, header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        raw_input = pd.read_csv(train,index_col=False)


        # TODO: Remember to implement the preprocess method

        train_dataset = self.preprocess(raw_input)
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)

        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y)

        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X_train[0])
        if not isinstance(self.y_train[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y_train[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X_train
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X_train), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X_train), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))

        self.train()
        error=self.predict()
        print('*********************************')
        #print(error)
        print("The total error on test data is " + str(np.sum(error)))
        print('*********************************')


    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #

    def __activation(self, x, activation):
        
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        elif activation == "tanh":
            self.__tanh(self, x)
        else:
            self.__ReLu(self, x)
            
    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        elif activation == "tanh":
            self.__tanh_derivative(self, x)
        else:
            self.__ReLu_derivative(self, x)            


    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __tanh(self, x):
        return (np.exp(x)-np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def __ReLu(self, x):
        return np.maximum(0, x)

    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def __tanh_derivative(self, x):
        return 1-x*x
    
    def __ReLu_derivative(self, x):
        return (x>0) * 1

    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #

    def preprocess(self, X):
        #categorical attributes to numerical values
        array=X.values
        attributes = array[:,0:-1]
        n = array[:,-1]
        labelencoder_N = LabelEncoder()
        n = labelencoder_N.fit_transform(n)

        # standardizing/scaling the attributes
        scaler = StandardScaler()
        scaler.fit(attributes)
        attributes = scaler.transform(attributes)
        

        X=pd.DataFrame(attributes)
        
        X.loc[:,4]=pd.Series(n,index=X.index)
        return X

    # Below is the training function

    def train(self, max_iterations = 1000, learning_rate = 0.7):
        print("Number of iterations: ", max_iterations)
        print("Learning rate: ", learning_rate)
        activation = "ReLu"
        print("Activation function applied: ", activation)
        for iteration in range(max_iterations):
            out = self.forward_pass(self.X_train)
            error = 0.5 * np.power((out - self.y_train), 2)
            self.backward_pass(out, activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations, the total training error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    def forward_pass(self,X):
        # pass our inputs through our neural network
        in1 = np.dot(X, self.w01 )
        self.X12 = self.__ReLu(in1)
        in2 = np.dot(self.X12, self.w12)
        self.X23 = self.__ReLu(in2)
        in3 = np.dot(self.X23, self.w23)
        out = self.__ReLu(in3)
        return out



    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        #print("Activation function applied:" , activation )
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    # TODO: Implement other activation functions

    def compute_output_delta(self, out, activation):
        if activation == "sigmoid":
            delta_output = (self.y_train - out) * (self.__sigmoid_derivative(out))
        elif activation == "tanh":
            delta_output = (self.y_train - out) * (self.__tanh_derivative(out))
        else:
            delta_output = (self.y_train - out) * (self.__ReLu_derivative(out))
        self.deltaOut = delta_output
        
        

    # TODO: Implement other activation functions

    def compute_hidden_layer2_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        elif activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
        else:
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__ReLu_derivative(self.X23))    
        self.delta23 = delta_hidden_layer2
        
        

    # TODO: Implement other activation functions

    def compute_hidden_layer1_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        elif activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12)) 
        else:
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__ReLu_derivative(self.X12))
        self.delta12 = delta_hidden_layer1


    
    # TODO: Implement other activation functions

    def compute_input_layer_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "tanh":
            delta_input_layer = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))    
        else:
            delta_input_layer = np.multiply(self.__ReLu_derivative(self.X01), self.delta01.dot(self.w01.T))
        self.delta01 = delta_input_layer
            
    
    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self):

        nrows = self.X_test.shape[1]
        for i in range(nrows):
            predicted=self.forward_pass(self.X_test)
        error = 0.5 * np.power((predicted - self.y_test), 2)

        return error


if __name__ == "__main__":
    neural_network = NeuralNet('iris.data')
    # neural_network.train()
    # testError = neural_network.predict("test.csv")
    # print(testError)
