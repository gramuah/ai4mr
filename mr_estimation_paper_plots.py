# Script to plot the figures included in the paper

import numpy as np
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV #To perform the search for the best parameters
from sklearn.metrics import mean_absolute_error #define the metrics to use for the evaluation
from matplotlib import pyplot
from joblib import dump, load

# load model (and test data) for further analysis

# select target (mass 'M' or radius 'R')
target = 'M'
#target = 'R'

# read models and test data

#Standard experiment
models = load('experiments/models_' + target + '.joblib')
X_test_norm, y_test, n_samples = load('experiments/test_data_' + target + '.joblib')


#Generalization experiment
#models = load('experiments/generalization_models_' + target + '.joblib')
#X_test_norm, y_test, n_samples = load('experiments/generalization_test_data_' + target + '.joblib')

sids = np.argsort(y_test)
y_test = y_test[sids]
X_test_norm = X_test_norm[sids, :]


# Select the model to use for the prediction to analyze the results
# 'nnet'-> MLPRegressor
# 'lr'-> LinearRegression
# 'dtr' -> DecisionTreeRegressor
# 'rf' ->RandomForestRegressor
# 'svm'->SVR
# 'bayes'->BayesianRidge
# 'knn'->KNeighborsRegressor
# 'stacking'->StackingRegressor

model_name = 'stacking'

# Repeat the prediction
if model_name == 'stacking':
    # NOTE: the stacking model can return the std for the prediction
    y_pred, y_std = models['stacking'].predict(X_test_norm, return_std=True)
else:
    y_pred = models[model_name].predict(X_test_norm)

# Figure 1
# Compare predictions with ground truth
score = mean_absolute_error(y_test, y_pred)
print("MAE: ", score)

reg_error = abs(y_test-y_pred)

n = np.arange(y_pred.size)

fig, ax = pyplot.subplots()
ax.scatter(n, y_test, c='tab:blue', label='y_test',alpha=0.3, edgecolors='none')
ax.scatter(n, y_pred, c='tab:orange', label='y_pred ('+model_name+')', alpha=0.3, edgecolors='none')
ax.legend()
ax.grid(True)
pyplot.ylabel(target)
pyplot.show()

#Figure with the error, ground truth, and predictions.
fig, ax = pyplot.subplots()
ax.plot(n, y_test, c='tab:blue', label='Test value', linewidth=3)
ax.plot(n, y_pred, c='tab:orange', label='Estimations', linewidth=3)
ax.plot(n, reg_error, c='tab:red', label='Absolute error', linewidth=3)

#show the std only for stacking
if model_name == 'stacking':
    ax.fill_between(n, y_pred-y_std, y_pred+y_std,color="pink", alpha=0.5, label="std")

ax.legend(fontsize=15)
ax.grid(True)
pyplot.ylabel(target + ' (in ' + target + 'o)',fontsize=15)
pyplot.xlabel('Star identifier', fontsize=15)
pyplot.yticks(fontsize=15)
pyplot.xticks(fontsize=15)
# Save figure
pyplot.savefig('results/results_ai_mr_' + target + '_' + model_name + '.pdf',  bbox_inches='tight')
pyplot.show()

