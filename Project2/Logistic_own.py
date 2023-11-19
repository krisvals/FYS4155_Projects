import autograd.numpy as np
import warnings
from copy import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
 
warnings.simplefilter("error")
warnings.filterwarnings('ignore', category=DeprecationWarning)
np.seterr(over='ignore')
 
def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

 
class LogisticRegression:
    def __init__(self, input_dim):
        self.weights = np.zeros((input_dim + 1, 1))  # Add bias term
 
    def fit(self, X, t, iter=1000, eta=0.01):
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term
 
        for i in range(iter):
                z = X @ self.weights
                predictions = sigmoid(z)
                gradient = X.T @ (predictions - t) / len(t)
                self.weights -= eta * gradient
                #Adding regularization
                self.weights += lmbd * self.weights
            
 
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term
        z = X @ self.weights
        return sigmoid(z)

        
def accuracy_score_numpy(Y_test, predictions):
    return np.sum(Y_test == predictions) / len(Y_test)
 
def train_test_split_numpy(inputs, labels, train_size, test_size):
    n_inputs = len(inputs)
    inputs_shuffled = inputs.copy()
    labels_shuffled = labels.copy()
    np.random.shuffle(inputs_shuffled)
    np.random.shuffle(labels_shuffled)
    train_end = int(n_inputs*train_size)
    X_train, X_test = inputs_shuffled[:train_end], inputs_shuffled[train_end:]
    Y_train, Y_test = labels_shuffled[:train_end], labels_shuffled[train_end:]
    return X_train, X_test, Y_train, Y_test


 
# Loading dataset
np.random.seed(0)
cancer = load_breast_cancer()
inputs = cancer.data
outputs = cancer.target

 
temp1 = np.reshape(inputs[:, 1], (len(inputs[:, 1]), 1))
temp2 = np.reshape(inputs[:, 2], (len(inputs[:, 2]), 1))
X = np.hstack((temp1, temp2))
temp = np.reshape(inputs[:, 5], (len(inputs[:, 5]), 1))
X = np.hstack((X, temp))
temp = np.reshape(inputs[:, 8], (len(inputs[:, 8]), 1))
X = np.hstack((X, temp))
 
y = outputs.reshape(outputs.shape[0], 1)
X_train, X_test, y_train, y_test = train_test_split_numpy(X, y, 0.7, 0.3)

#Hyperparameters to adjust
lmbd_vals = [1e-3, 1e-2, 1e-1, 1]
#eta_vals = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
iterations = [1,10,100,1000]

train_accuracy = np.zeros((len(lmbd_vals), len(iterations)))
test_accuracy = np.zeros((len(lmbd_vals), len(iterations)))

for j, lmbd in enumerate(lmbd_vals):
    for i, iteration in enumerate(iterations): # starts a loop to test variables.
    	logreg_model = LogisticRegression(input_dim=X_train.shape[1])
    	logreg_model.fit(X_train, y_train, iter=iteration, eta= 1e-4)
    	train_predictions = logreg_model.predict(X_train)
    	test_predictions = logreg_model.predict(X_test)
    
    	score_train = accuracy_score_numpy(y_train, train_predictions) 
    	score_test = accuracy_score_numpy(y_test, test_predictions)
    
    	train_accuracy[i][j] = score_train
    	test_accuracy[i][j] = score_test

 
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

fontsize = 12   

train_accuracy *= 100
test_accuracy *= 100

fig, ax = plt.subplots(figsize = (6, 6))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis", xticklabels=iterations, yticklabels=lmbd_vals, fmt=".1f",  annot_kws={"size": fontsize},  vmin=0, vmax=100)
ax.set_title("Training Accuracy Own Code [%]", fontsize=fontsize)
ax.set_ylabel("Regularization Lambda", fontsize=fontsize)
ax.set_xlabel("Iterations", fontsize=fontsize)
plt.tick_params(axis='both', labelsize=fontsize)
plt.savefig('/home/Kristen/Documents/FYS4155/Excecises/C_own_Lambda_Train.pdf', format='pdf', bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize = (6, 6))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis", xticklabels=iterations, yticklabels=lmbd_vals, fmt=".1f",  annot_kws={"size": fontsize},  vmin=0, vmax=100)
ax.set_title("Test Accuracy Own Code [%]", fontsize=fontsize)
ax.set_ylabel("Regularization Lambda", fontsize=fontsize)
ax.set_xlabel("Iterations", fontsize=fontsize)
plt.tick_params(axis='both', labelsize=fontsize)
plt.savefig('/home/Kristen/Documents/FYS4155/Excecises/C_own_Lambda_Test.pdf', format='pdf', bbox_inches="tight")
plt.show()
