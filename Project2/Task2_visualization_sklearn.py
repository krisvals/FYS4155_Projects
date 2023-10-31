#Visualization sklearn

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()

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

n = 1000 #number of datapoints
xlim = 1.0
np.random.seed(42)
x = xlim*np.random.rand(n,1)
x.sort(axis = 0)

# Design matrix
xnew = []
deg = 2
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

M = 10
lmbd_vals = [1e-20, 1e-15, 1e-10, 1e-5, 1e0]
eta_vals = [1e-20, 1e-16, 1e-14, 1e-10, 1e-6] #Can be adjusted, but code gets very slow for eta below 1e-10
epochs = 100000

DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

# grid search
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = MLPRegressor(
            hidden_layer_sizes=(50,), #number of neurons and hidden layers
            activation='logistic', #logistic = sigmoid
            learning_rate_init=eta,
            alpha=lmbd,
            batch_size = M,
            beta_1=0.9,  
            beta_2=0.999, 
            early_stopping=True,
            epsilon=1e-08,
            max_iter=epochs,
            momentum=0.1778,
            nesterovs_momentum=True,
            random_state=None,
            shuffle=False,
            solver='sgd',
            tol=1E-6,
            verbose=False,
            warm_start=False
        )
        dnn.fit(X_train, y_train)
        DNN_numpy[i][j] = dnn
        test_predict = dnn.predict(X_test)
        
        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("Accuracy score on test set: ", r2_score(y_test, test_predict))

for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        dnn = DNN_numpy[i][j]
        train_pred = dnn.predict(X_train) 
        test_pred = dnn.predict(X_test)
        train_accuracy[i][j] = mean_squared_error(y_train, train_pred)
        test_accuracy[i][j] = mean_squared_error(y_test, test_pred)

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis_r", xticklabels=lmbd_vals, yticklabels=eta_vals)
ax.set_title("Training Accuracy")
ax.set_ylabel("eta")
ax.set_xlabel("lambda")
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis_r", xticklabels=lmbd_vals, yticklabels=eta_vals)
ax.set_title("Test Accuracy")
ax.set_ylabel("eta")
ax.set_xlabel("lambda")
plt.show()
