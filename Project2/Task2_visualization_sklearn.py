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
