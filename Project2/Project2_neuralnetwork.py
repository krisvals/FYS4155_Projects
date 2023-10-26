# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:55:42 2023

@author: erlendou
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score
 
from sklearn.model_selection import train_test_split

def MSE(y,ypred):
        
    MSE = 0
   
    for i in range(len(y)):
        MSE += (y[i]-ypred[i])**2
             
    MSE /= len(y)
  
    return(MSE/np.mean(abs(y))*100)

def FrankeFunction(x):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) )
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 )
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 )
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 )
    return term1 + term2 + term3 + term4

def F(x):
    a0  = random.randint(-10,10)
    a1 = random.randint(-10,10)
    a2 = random.randint(-10,10)
    return(a0 + a1*x + a2*x**2)

# ensure the same random numbers appear every time
n = 1000 #number of datapoints
xlim = 1.0
np.random.seed(42)
x = xlim*np.random.rand(n,1)
x.sort(axis = 0)


# Design matrix
xnew = []
deg = 1
for kk in range(n):
    xnew.append([])
    xnew[kk].append(1)
    for jj in range(deg):
        xnew[kk].append(float(x[kk]**(jj+1)))
X =  np.c_[xnew] 


# Franke
y = FrankeFunction(x).reshape(n, )


y=y.astype('float')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Defining the neural network
n_inputs, n_features = X.shape
n_hidden_neurons = 10
n_categories = 20
n_features = 20
M = int(n/200)
eta_vals = np.logspace(-10, .1, 7)
lmbd_vals = np.logspace(-9, -.99, 1)
# store models for later use
DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
epochs = 1000

from sklearn.neural_network import MLPClassifier
# store models for later use
#DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
  
scorelist =  []
varlist = []
MSElist = []
#sklearn neural network (I think?)
for m in lmbd_vals:
    
    varlist.append(m)
    dnn = (MLPRegressor(activation='logistic', alpha=0.0001, batch_size=M, beta_1=0.9,
                        beta_2=0.999, early_stopping=False, epsilon=1e-08,
                        hidden_layer_sizes=n_hidden_neurons, learning_rate='constant',
                        max_iter=1000, momentum=0.01,
                        nesterovs_momentum=True, random_state=None,
                        shuffle=True, solver='lbfgs', tol=0.00001, verbose=False, warm_start=False))
    dnn.fit(X_train, y_train)
    ypred = dnn.predict(X)
    scorelist.append(dnn.score(X_test, y_test))
    MSElist.append(MSE(y, ypred))

#plt.plot(x,ypred, 'ro', alpha = 0.15)      
#plt.plot(x,y, 'b-')
#DNN_scikit[i][j] = dnn
plt.plot(np.log10(varlist),MSElist)  
       
print("Accuracy score on test set: ", dnn.score(X_test, y_test))
print()


