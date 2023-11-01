# Grid search and visualization own code
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
sns.set()

M = 10
lmbd_vals = [1e-20, 1e-15, 1e-10, 1e-5, 1e0] 
eta_vals = [1e-20, 1e-16, 1e-14, 1e-10, 1e-6] #Can be adjusted, but code gets very slow for eta below 1e-10
epochs = 1000000

DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

# grid search
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        FFNN1.reset_weights() #important to reset weights, as to not generate data based on previous runs
        scores = FFNN1.fit(X_train,y_train, Adagrad(eta = eta), epochs = 1000) #train data to the fit. Using the training data here, not sure if this is the best option. Many parameters can be set to optimize the fit, such as eta, lambda, momentum.   
        DNN_numpy[i][j] = FFNN1
        test_predict = FFNN1.predict(X) #creates a prediction from the training.
        score = FFNN1._accuracy(y,test_predict)
        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("Accuracy score on test set: ", mean_squared_error(y, test_predict))

train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

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
