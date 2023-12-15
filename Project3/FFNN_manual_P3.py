# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:30:35 2023

@author: erlendou
"""

import math
import autograd.numpy as np
import sys
import warnings
from autograd import grad, elementwise_grad
from random import random, seed
from copy import deepcopy, copy
from typing import Tuple, Callable
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns


warnings.simplefilter("error")
warnings.filterwarnings('ignore', category=DeprecationWarning)

def CostOLS(target):
    
    def func(X):
        return (1.0 / target.shape[0]) * np.sum((target - X) ** 2)

    return func


def CostLogReg(target):

    def func(X):
        
        return -(1.0 / target.shape[0]) * np.sum(
            (target * np.log(X + 10e-10)) + ((1 - target) * np.log(1 - X + 10e-10))
        )

    return func


def CostCrossEntropy(target):
    
    def func(X):
        return -(1.0 / target.size) * np.sum(target * np.log(X + 10e-10))

    return func




def identity(X):
    return X


def sigmoid(X):
    try:
        return 1.0 / (1 + np.exp(-X))
    except FloatingPointError:
        return np.where(X > np.zeros(X.shape), np.ones(X.shape), np.zeros(X.shape))


def softmax(X):
    X = X - np.max(X, axis=-1, keepdims=True)
    delta = 10e-10
    return np.exp(X) / (np.sum(np.exp(X), axis=-1, keepdims=True) + delta)


def RELU(X):
    return np.where(X > np.zeros(X.shape), X, np.zeros(X.shape))


def LRELU(X):
    delta = 10e-4
    return np.where(X > np.zeros(X.shape), X, delta * X)


def derivate(func):
    if func.__name__ == "RELU":

        def func(X):
            return np.where(X > 0, 1, 0)

        return func

    elif func.__name__ == "LRELU":

        def func(X):
            delta = 10e-4
            return np.where(X > 0, 1, delta)

        return func

    else:
        return elementwise_grad(func)

import autograd.numpy as np

class Scheduler:
    """
    Abstract class for Schedulers
    """

    def __init__(self, eta):
        self.eta = eta

    # should be overwritten
    def update_change(self, gradient):
        raise NotImplementedError

    # overwritten if needed
    def reset(self):
        pass


class Constant(Scheduler):
    def __init__(self, eta):
        super().__init__(eta)

    def update_change(self, gradient):
        return self.eta * gradient
    
    def reset(self):
        pass


class Momentum(Scheduler):
    def __init__(self, eta: float, momentum: float):
        super().__init__(eta)
        self.momentum = momentum
        self.change = 0

    def update_change(self, gradient):
        self.change = self.momentum * self.change + self.eta * gradient
        return self.change

    def reset(self):
        pass


class Adagrad(Scheduler):
    def __init__(self, eta):
        super().__init__(eta)
        self.G_t = None

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero

        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        self.G_t += gradient @ gradient.T

        G_t_inverse = 1 / (
            delta + np.sqrt(np.reshape(np.diagonal(self.G_t), (self.G_t.shape[0], 1)))
        )
        return self.eta * gradient * G_t_inverse

    def reset(self):
        self.G_t = None


class AdagradMomentum(Scheduler):
    def __init__(self, eta, momentum):
        super().__init__(eta)
        self.G_t = None
        self.momentum = momentum
        self.change = 0

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero

        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        self.G_t += gradient @ gradient.T

        G_t_inverse = 1 / (
            delta + np.sqrt(np.reshape(np.diagonal(self.G_t), (self.G_t.shape[0], 1)))
        )
        self.change = self.change * self.momentum + self.eta * gradient * G_t_inverse
        return self.change

    def reset(self):
        self.G_t = None


class RMS_prop(Scheduler):
    def __init__(self, eta, rho):
        super().__init__(eta)
        self.rho = rho
        self.second = 0.0

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero
        self.second = self.rho * self.second + (1 - self.rho) * gradient * gradient
        return self.eta * gradient / (np.sqrt(self.second + delta))

    def reset(self):
        self.second = 0.0


class Adam(Scheduler):
    def __init__(self, eta, rho, rho2):
        super().__init__(eta)
        self.rho = rho
        self.rho2 = rho2
        self.moment = 0
        self.second = 0
        self.n_epochs = 1

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero

        self.moment = self.rho * self.moment + (1 - self.rho) * gradient
        self.second = self.rho2 * self.second + (1 - self.rho2) * gradient * gradient

        moment_corrected = self.moment / (1 - self.rho**self.n_epochs)
        second_corrected = self.second / (1 - self.rho2**self.n_epochs)

        return self.eta * moment_corrected / (np.sqrt(second_corrected + delta))

    def reset(self):
        self.n_epochs += 1
        self.moment = 0
        self.second = 0
momentum_scheduler = Momentum(eta=1e-3, momentum=0.9)
adam_scheduler = Adam(eta=1e-3, rho=0.9, rho2=0.999)
class FFNN:
    """
    Description:
    ------------
        Feed Forward Neural Network with interface enabling flexible design of a
        nerual networks architecture and the specification of activation function
        in the hidden layers and output layer respectively. This model can be used
        for both regression and classification problems, depending on the output function.

    Attributes:
    ------------
        I   dimensions (tuple[int]): A list of positive integers, which specifies the
            number of nodes in each of the networks layers. The first integer in the array
            defines the number of nodes in the input layer, the second integer defines number
            of nodes in the first hidden layer and so on until the last number, which
            specifies the number of nodes in the output layer.
        II  hidden_func (Callable): The activation function for the hidden layers
        III output_func (Callable): The activation function for the output layer
        IV  cost_func (Callable): Our cost function
        V   seed (int): Sets random seed, makes results reproducible
    """

    def __init__(
        self,
        dimensions: tuple[int],
        hidden_func: Callable = sigmoid,
        output_func: Callable = lambda x: x,
        cost_func: Callable = CostOLS,
        seed: int = None,
    ):
        self.dimensions = dimensions
        self.hidden_func = hidden_func
        self.output_func = output_func
        self.cost_func = cost_func
        self.seed = seed
        self.weights = list()
        self.schedulers_weight = list()
        self.schedulers_bias = list()
        self.a_matrices = list()
        self.z_matrices = list()
        self.classification = None

        self.reset_weights()
        self._set_classification()

    def fit(
        self,
        X: np.ndarray,
        t: np.ndarray,
        scheduler: Scheduler,
        batches: int = 1,
        epochs: int = 100,
        lam: float = 0,
        X_val: np.ndarray = None,
        t_val: np.ndarray = None,
    ):
        """
        Description:
        ------------
            This function performs the training the neural network by performing the feedforward and backpropagation
            algorithm to update the networks weights.

        Parameters:
        ------------
            I    X (np.ndarray) : training data
            II   t (np.ndarray) : target data
            III  scheduler (Scheduler) : specified scheduler (algorithm for optimization of gradient descent)
            IV   scheduler_args (list[int]) : list of all arguments necessary for scheduler

        Optional Parameters:
        ------------
            V    batches (int) : number of batches the datasets are split into, default equal to 1
            VI   epochs (int) : number of iterations used to train the network, default equal to 100
            VII  lam (float) : regularization hyperparameter lambda
            VIII X_val (np.ndarray) : validation set
            IX   t_val (np.ndarray) : validation target set

        Returns:
        ------------
            I   scores (dict) : A dictionary containing the performance metrics of the model.
                The number of the metrics depends on the parameters passed to the fit-function.

        """

        # setup 
        if self.seed is not None:
            np.random.seed(self.seed)

        val_set = False
        if X_val is not None and t_val is not None:
            val_set = True

        # creating arrays for score metrics
        train_errors = np.empty(epochs)
        train_errors.fill(np.nan)
        val_errors = np.empty(epochs)
        val_errors.fill(np.nan)

        train_accs = np.empty(epochs)
        train_accs.fill(np.nan)
        val_accs = np.empty(epochs)
        val_accs.fill(np.nan)

        self.schedulers_weight = list()
        self.schedulers_bias = list()

        batch_size = int(X.shape[0] // batches)
        
        X, t = resample(X, t)

        # this function returns a function valued only at X
        cost_function_train = self.cost_func(t)
        if val_set:
            cost_function_val = self.cost_func(t_val)

        # create schedulers for each weight matrix
        for i in range(len(self.weights)):
            self.schedulers_weight.append(copy(scheduler))
            self.schedulers_bias.append(copy(scheduler))

        #print(f"{scheduler.__class__.__name__}: Eta={scheduler.eta}, Lambda={lam}")

        try:
            for e in range(epochs):
                
                for i in range(batches):
                    # allows for minibatch gradient descent
                    if i == batches - 1:
                        # If the for loop has reached the last batch, take all thats left
                        X_batch = X[i * batch_size :, :]
                        t_batch = t[i * batch_size :, :]
                    else:
                        X_batch = X[i * batch_size : (i + 1) * batch_size, :]
                        t_batch = t[i * batch_size : (i + 1) * batch_size, :]

                    self._feedforward(X_batch)
                    self._backpropagate(X_batch, t_batch, lam)

                # reset schedulers for each epoch (some schedulers pass in this call)
                for scheduler in self.schedulers_weight:
                    scheduler.reset()

                for scheduler in self.schedulers_bias:
                    scheduler.reset()

                # computing performance metrics
                pred_train = self.predict(X)
                
                train_error = cost_function_train(pred_train)
                

                train_errors[e] = train_error
                if(abs(train_errors[e]-train_errors[e-1]) <= 1E-8):
                    print(e)
                    break;
                    
                if val_set:
                    
                    pred_val = self.predict(X_val)
                    val_error = cost_function_val(pred_val)
                    val_errors[e] = val_error

                if self.classification:
                    train_acc = self._accuracy(self.predict(X), t)
                    train_accs[e] = train_acc
                    if val_set:
                        val_acc = self._accuracy(pred_val, t_val)
                        val_accs[e] = val_acc

                # printing progress bar
                #progression = e / epochs
                #print_length = self._progress_bar(
                    #progression,
                    #train_error=train_errors[e],
                    #train_acc=train_accs[e],
                    #val_error=val_errors[e],
                    #val_acc=val_accs[e],
                #)
        except KeyboardInterrupt:
            # allows for stopping training at any point and seeing the result
            pass

        # visualization of training progression (similiar to tensorflow progression bar)
        #sys.stdout.write("\r" + " " * print_length)
        #sys.stdout.flush()
        self._progress_bar(
            1,
            train_error=train_errors[e],
            train_acc=train_accs[e],
            val_error=val_errors[e],
            val_acc=val_accs[e],
        )
        #sys.stdout.write("")

        # return performance metrics for the entire run
        scores = dict()

        scores["train_errors"] = train_errors

        if val_set:
            scores["val_errors"] = val_errors

        if self.classification:
            scores["train_accs"] = train_accs

            if val_set:
                scores["val_accs"] = val_accs

        return scores

    def predict(self, X: np.ndarray, *, threshold=0.5):
        """
         Description:
         ------------
             Performs prediction after training of the network has been finished.

         Parameters:
        ------------
             I   X (np.ndarray): The design matrix, with n rows of p features each

         Optional Parameters:
         ------------
             II  threshold (float) : sets minimal value for a prediction to be predicted as the positive class
                 in classification problems

         Returns:
         ------------
             I   z (np.ndarray): A prediction vector (row) for each row in our design matrix
                 This vector is thresholded if regression=False, meaning that classification results
                 in a vector of 1s and 0s, while regressions in an array of decimal numbers

        """

        predict = self._feedforward(X)

        if self.classification:
            return np.where(predict > threshold, 1, 0)
        else:
            return predict

    def reset_weights(self):
        """
        Description:
        ------------
            Resets/Reinitializes the weights in order to train the network for a new problem.

        """
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = list()
        for i in range(len(self.dimensions) - 1):
            weight_array = np.random.randn(
                self.dimensions[i] + 1, self.dimensions[i + 1]
            )
            weight_array[0, :] = np.random.randn(self.dimensions[i + 1]) * 0.01

            self.weights.append(weight_array)

    def _feedforward(self, X: np.ndarray):
        """
        Description:
        ------------
            Calculates the activation of each layer starting at the input and ending at the output.
            Each following activation is calculated from a weighted sum of each of the preceeding
            activations (except in the case of the input layer).

        Parameters:
        ------------
            I   X (np.ndarray): The design matrix, with n rows of p features each

        Returns:
        ------------
            I   z (np.ndarray): A prediction vector (row) for each row in our design matrix
        """

        # reset matrices
        self.a_matrices = list()
        self.z_matrices = list()

        # if X is just a vector, make it into a matrix
        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))

        # Add a coloumn of zeros as the first coloumn of the design matrix, in order
        # to add bias to our data
        bias = np.ones((X.shape[0], 1)) * 0.01
        X = np.hstack([bias, X])

        # a^0, the nodes in the input layer (one a^0 for each row in X - where the
        # exponent indicates layer number).
        a = X
        self.a_matrices.append(a)
        self.z_matrices.append(a)

        # The feed forward algorithm
        for i in range(len(self.weights)):
            if i < len(self.weights) - 1:
                #print(a)
                z = a @ self.weights[i]
                self.z_matrices.append(z)
                a = self.hidden_func(z)
                # bias column again added to the data here
                bias = np.ones((a.shape[0], 1)) * 0.01
                a = np.hstack([bias, a])
                self.a_matrices.append(a)
            else:
                try:
                    # a^L, the nodes in our output layers
                    z = a @ self.weights[i]
                    a = self.output_func(z)
                    self.a_matrices.append(a)
                    self.z_matrices.append(z)
                except Exception as OverflowError:
                    print(
                        "OverflowError in fit() in FFNN\nHOW TO DEBUG ERROR: Consider lowering your learning rate or scheduler specific parameters such as momentum, or check if your input values need scaling"
                    )

        # this will be a^L
        return a

    def _backpropagate(self, X, t, lam):
        """
        Description:
        ------------
            Performs the backpropagation algorithm. In other words, this method
            calculates the gradient of all the layers starting at the
            output layer, and moving from right to left accumulates the gradient until
            the input layer is reached. Each layers respective weights are updated while
            the algorithm propagates backwards from the output layer (auto-differentation in reverse mode).

        Parameters:
        ------------
            I   X (np.ndarray): The design matrix, with n rows of p features each.
            II  t (np.ndarray): The target vector, with n rows of p targets.
            III lam (float32): regularization parameter used to punish the weights in case of overfitting

        Returns:
        ------------
            No return value.

        """
        out_derivative = derivate(self.output_func)
        hidden_derivative = derivate(self.hidden_func)

        for i in range(len(self.weights) - 1, -1, -1):
            # delta terms for output
            if i == len(self.weights) - 1:
                # for multi-class classification
                if (
                    self.output_func.__name__ == "softmax"
                ):
                    delta_matrix = self.a_matrices[i + 1] - t
                # for single class classification
                else:
                    cost_func_derivative = grad(self.cost_func(t))
                    delta_matrix = out_derivative(
                        self.z_matrices[i + 1]
                    ) * cost_func_derivative(self.a_matrices[i + 1])

            # delta terms for hidden layer
            else:
                delta_matrix = (
                    self.weights[i + 1][1:, :] @ delta_matrix.T
                ).T * hidden_derivative(self.z_matrices[i + 1])

            # calculate gradient
            gradient_weights = self.a_matrices[i][:, 1:].T @ delta_matrix
            gradient_bias = np.sum(delta_matrix, axis=0).reshape(
                1, delta_matrix.shape[1]
            )

            # regularization term
            gradient_weights += self.weights[i][1:, :] * lam

            # use scheduler
            update_matrix = np.vstack(
                [
                    self.schedulers_bias[i].update_change(gradient_bias),
                    self.schedulers_weight[i].update_change(gradient_weights),
                ]
            )

            # update weights and bias
            self.weights[i] -= update_matrix

    def _accuracy(self, prediction: np.ndarray, target: np.ndarray):
        """
        Description:
        ------------
            Calculates accuracy of given prediction to target

        Parameters:
        ------------
            I   prediction (np.ndarray): vector of predicitons output network
                (1s and 0s in case of classification, and real numbers in case of regression)
            II  target (np.ndarray): vector of true values (What the network ideally should predict)

        Returns:
        ------------
            A floating point number representing the percentage of correctly classified instances.
        """
        print(1)
        assert prediction.size == target.size
        return np.average((target == prediction))
    def _set_classification(self):
        """
        Description:
        ------------
            Decides if FFNN acts as classifier (True) og regressor (False),
            sets self.classification during init()
        """
        self.classification = False
        if (
            self.cost_func.__name__ == "CostLogReg"
            or self.cost_func.__name__ == "CostCrossEntropy"
        ):
            self.classification = True

    def _progress_bar(self, progression, **kwargs):
        """
        Description:
        ------------
            Displays progress of training
        """
        print_length = 40
        num_equals = int(progression * print_length)
        num_not = print_length - num_equals
        arrow = ">" if num_equals > 0 else ""
        bar = "[" + "=" * (num_equals - 1) + arrow + "-" * num_not + "]"
        perc_print = self._format(progression * 100, decimals=5)
        #line = f"  {bar} {perc_#}% "

        for key in kwargs:
            if not np.isnan(kwargs[key]):
                value = self._format(kwargs[key], decimals=4)
                #line += f"| {key}: {value} "
        #sys.stdout.write("\r" + line)
        #sys.stdout.flush()
        #return len(line)

    def _format(self, value, decimals=4):
        """
        Description:
        ------------
            Formats decimal numbers for progress bar
        """
        if value > 0:
            v = value
        elif value < 0:
            v = -10 * value
        else:
            v = 1
        n = 1 + math.floor(math.log10(v))
        if n >= decimals - 1:
            return str(round(value))
        return f"{value:.{decimals-n-1}f}"



#Generating design matrix
def createX(x):
    X = np.zeros(len(x))
    for i in range(len(x)):
        X[i] = x[i]
    return(X)


#Returns mean squared error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

#Returns accuracy score, although this one is mostly 0
def accuracy_score_numpy(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)

#function to load data
def loadData(fileName):
    days = []
    date = []
    streamflow = []
    precip = []
    temp_min = []
    temp_max = []
    humidity = []
    radiation = []
    
    f = open(fileName, 'r')
    f.readline()
    day = 0
    for line in f:
        cline = line.split(',')
        days.append(day)
        date.append(str(cline[1]))
        streamflow.append(float(cline[2]))
        precip.append(float(cline[3]))
        temp_min.append(float(cline[4]))
        temp_max.append(float(cline[5]))
        humidity.append(float(cline[6]))
        radiation.append(float(cline[7]))
        day += 1
    return(days, date,streamflow,precip, temp_min, temp_max,humidity,radiation)

def createDim(n_in,n_hidden_layer,n_hidden_nodes,n_ou):
    dimensions = []
    dimensions.append(n_in)
    hidden = n_hidden_layer*[n_hidden_nodes]
    for i in hidden:
        dimensions.append(i)
    dimensions.append(n_ou)
    return(dimensions)     
 
#main
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#load data from file
print("Start")
days, date,streamflow,precip, temp_min, temp_max,humidity,radiation = loadData("Project3_data.csv")
print("Data loaded")
#initialize parameters. Might need some normalization.
#create design matrix and y
precip = precip / np.mean(precip)
precip = createX(precip).reshape(-1,1)
temp_min = temp_min / np.mean(temp_min)
temp_min = createX(temp_min).reshape(-1,1)

temp_max = temp_max / np.mean(temp_max)
temp_max = createX(temp_max).reshape(-1,1)

humidity = humidity / np.mean(humidity)
humidity = createX(humidity).reshape(-1,1)

radiation = radiation / np.mean(radiation)
radiation = createX(radiation).reshape(-1,1)


#x = radiation
x = np.concatenate((humidity, radiation, temp_max, temp_min), axis=1)


y = streamflow/np.mean(streamflow)
y = createX(streamflow).reshape(-1,1)

print("Mean" + str(np.mean(streamflow)))
#y = scaler.fit_transform(y.reshape(-1,1)).ravel()

np.random.seed(42) #creates random seed to have reproducible results

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y)

n_output_nodes = 1 #s et to 1 for regression
n_input_nodes = X_train.shape[1] # set number of input nodes



MSElist = [] #list of MSE. Not sure if this is usefull.
n_hidden_nodes = 3 # set number of hidden nodes
n_hidden_layers = 1 # set number of hidden layers
dimensions = createDim(n_input_nodes,n_hidden_layers,n_hidden_nodes,n_output_nodes)
print(dimensions)
FFNN1 = FFNN(dimensions = dimensions,hidden_func = sigmoid ,  seed = 42) #creates an object of the neural network
FFNN1.reset_weights() #reset weights, as to start fresh every time
M = int(50)

hyper1 =  [1E-5,1E-4,1E-3,1E-2]
hyper2 = [1E-2,0.1,0.5,0.8,0.9] 
epochs = 100


train_accuracy = np.zeros((len(hyper2), len(hyper1)))
test_accuracy = np.zeros((len(hyper2), len(hyper1)))


for i, h1 in enumerate(hyper1):     
    for j, h2 in enumerate(hyper2): # starts a loop to test variables.
        n_hidden_layers = 10
        n_hidden_nodes = 120
        dimensions = createDim(n_input_nodes,n_hidden_layers,n_hidden_nodes,n_output_nodes)
        FFNN1 = FFNN(dimensions = dimensions,hidden_func = sigmoid ,  seed = 42) #creates an object of the neural network
               
        FFNN1.reset_weights() #important to reset weights, as to not generate data based on previous runs
        FFNN1.fit(X_train,y_train, AdagradMomentum(h1,h2), epochs = int(200), lam = 1E-6, batches = int(50)) #train data to the fit. Using the training data here, not sure if this is the best option. Many parameters can be set to optimize the fit, such as eta, lambda, momentum.
        
        train_pred = FFNN1.predict(X_train) #creates a prediction from the training. Hopefully this is working, but not completely sure.
        test_pred = FFNN1.predict(X_test)
        plot_pred = FFNN1.predict(x)
       

        MSE_train = mean_squared_error(y_train, train_pred) #return MSE of the fit.
        MSE_test = mean_squared_error(y_test, test_pred)
        
        train_accuracy[j][i] = MSE_train
        test_accuracy[j][i] = MSE_test
        MSElist.append(MSE_train)
        print("batch size = ", h2)
        print("learning rate = ", h1)
        print("MSE score of training data: ", MSE_train)
        print("MSE score of test data: ", MSE_test)
        plt.plot(days,y, 'b-', linewidth = 4, label = 'data')
        plt.plot(days,plot_pred, 'r--', linewidth = 3, label = 'fit')
        plt.title("Daily streamflow")
        plt.xlabel('Day', fontsize = 18)
        plt.ylabel('Streamflow', fontsize = 18)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.legend()
        #plt.savefig('FinalFit_4variable .pdf')
        plt.show()
fnt = 18 
"""
for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        dnn = DNN_numpy[i][j]
        train_pred = dnn.predict(X_train) 
        test_pred = dnn.predict(X_test)
        train_accuracy[i][j] = mean_squared_error(y_train, train_pred)
        test_accuracy[i][j] = mean_squared_error(y_test, test_pred)
"""

#plt.plot(days,y)
#plt.plot(days,plot_pred)
#plt.show()



fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis_r", xticklabels=hyper1, yticklabels=hyper2)
ax.set_title("MSE for Adagrad learning schedule with momentum", fontsize = fnt-4)
ax.set_xticklabels(hyper1,fontsize = fnt)
ax.set_yticklabels(hyper2,fontsize = fnt)
ax.set_xlabel("Learning rate", fontsize = fnt)
ax.set_ylabel("Momentum", fontsize = fnt)

plt.savefig("MSE_AdagradMomentumh2_epochs200_lam-6_M50_etah1_4v.pdf")
plt.show()

"""
plt.plot(np.log10(hyper1), MSElist, linewidth = 6)
plt.xticks(fontsize = fnt)
plt.yticks(fontsize = fnt)
plt.xlabel("log(n epochs)", fontsize = fnt)
plt.ylabel("MSE", fontsize = fnt)
plt.title("MSE as a function of number of epochs", fontsize = fnt)
plt.savefig("MSE_AdagradMomentum_epochs.pdf")
"""
"""

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis_r", xticklabels=lmbd_vals, yticklabels=eta_vals)
ax.set_title("MSE for Adam with $\eta$ = 1E-2", fontsize =fnt)
ax.set_xticklabels(lmbd_vals,fontsize = fnt)
ax.set_yticklabels(eta_vals,fontsize = fnt)

ax.set_ylabel("rho", fontsize = fnt)
ax.set_xlabel("rho2", fontsize = fnt)
plt.show()



#print(scores) #prints scores, although not sure how to use them yet
plt.plot(x,y, '--',label = 'data', linewidth = 10) #plot data
plt.plot(x,plot_pred, 'r', linewidth = 3, label = 'fit', alpha = 1 ) #plot fit
plt.xlabel('x', fontsize = fnt)
plt.ylabel('y', fontsize = fnt)
plt.title('Fit of the Franke Function', fontsize = fnt)
plt.xticks(fontsize = fnt-6)
plt.yticks(fontsize = fnt-4)
plt.legend()
plt.savefig('FrankeFit.pdf')
plt.show()
"""
#plt.plot(np.log10(varlist),MSElist) #plot MSE for parameter testing

