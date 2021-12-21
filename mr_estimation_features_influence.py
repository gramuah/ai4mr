# Stellar mass and radius estimation
# Experiment to analyze the influence of the features in the regressor.
# We explore how an incremental incoporation of features affects to the performance of the final model

from numpy import mean
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV  # To perform the search for the best parameters
from sklearn.metrics import mean_absolute_error  # define the metric to use for the evaluation
from matplotlib import pyplot
from joblib import dump, load
import pandas as pd
import numpy as np


# this cript imports the data with the information of the stars
def get_dataset():
    data = pd.read_table('data/data_sample_mass_radius.txt', sep="\t")
    # read data with errors
    df = data[
        ['R', 'eR1', 'eR2', 'M', 'eM1', 'eM2', 'Teff', 'eTeff1', 'eTeff2', 'logg', 'elogg1', 'elogg2', 'Meta', 'eMeta1',
         'eMeta2', 'L',
         'eL1', 'eL2']]

    # clean NA values (simply remove the corresponding columns)
    df.dropna(inplace=True, axis=0)
    return df


# augment data using uncertainties
def data_augmentation_with_uncertainties(X_input, y_input, n_samples):
    # Generate samples for every input point using a uniform distribution

    # Input data comes in the form: X_input ( feat0, efeat0_1, efeat0_2, feat1, efeat1_1, efeat1_2, ...)
    # that is, every feature is followed by two error bounds (lower and upper)

    # Separate data and errors

    # Input data used for the experiments, example
    # X_input['Teff', 'eTeff1', 'eTeff2', 'logg', 'elogg1', 'elogg2', 'Meta', 'eMeta1', 'eMeta2', 'L', 'eL1', 'eL2']]
    # y_input['M, 'eM1', 'eM2'] or y['R, 'eR1', 'eR2']

    # read features

    X = X_input[:, 0::3]

    # read errors
    m, n = X_input.shape
    num_features = int(n / 3)
    Xe = np.empty((m, num_features * 2), float)
    jj = 0
    kk = 0
    for ii in range(num_features):
        Xe[:, kk] = X_input[:, 1 + jj]
        kk += 1
        Xe[:, kk] = X_input[:, 2 + jj]
        kk += 1
        jj = jj + 3

    # repeat for the target variable
    y = y_input[:, 0::3]
    ye = y_input[:, 1::]

    if n_samples == 0:  # no random sampling is needed, return original data without error bounds
        return X, np.ravel(y)

    # Initialize random number generator
    from numpy.random import default_rng

    seed = 1
    rng = default_rng(seed)

    first = True
    jj = 0
    # iterate over the arrays
    for (s_x, s_xe, s_y, s_ye) in zip(X, Xe, y, ye):
        # generate new samples
        y_new = rng.uniform(s_y - s_ye[0], s_y + s_ye[1], (n_samples, 1))

        X_new = np.empty((n_samples, num_features), float)
        ee = 0
        for ff in range(num_features):
            new_sample = rng.uniform(s_x[ff] - s_xe[ee+0], s_x[ff] + s_xe[ee+1], (1, n_samples))
            X_new[:,ff] = new_sample
            ee = ee + 2

        if first:  # to initialize aug variables
            y_aug = np.vstack((y[0, :], y_new))
            X_aug = np.vstack((X[0, :], X_new))
            first = False
        else:
            y_aug = np.vstack((y_aug, y[jj, :], y_new))
            X_aug = np.vstack((X_aug, X[jj, :], X_new))
        jj += 1

    return X_aug, np.ravel(y_aug)


# get a stacking ensemble of models using the best selection
def get_best_stacking():
    # define the level 0 models
    # the best combination is as follows
    level0 = list()

    level0.append(('nnet',
                   MLPRegressor(activation='relu', hidden_layer_sizes=(25, 25, 25, 25), learning_rate='adaptive',
                                learning_rate_init=0.2, max_iter=1000, solver='sgd', alpha=0.01,
                                random_state=0, verbose=False)))

    level0.append(('rf', RandomForestRegressor(random_state=0)))
    level0.append(('knn', KNeighborsRegressor()))

    # define meta learner model
    level1 = BayesianRidge()

    # define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    return model


# get a list of models to evaluate
def get_models():
    models = dict()

    # LinearRegression
    models['lr'] = LinearRegression()

    # DecisionTree
    # parameter to optimize in the decision tree
    tuned_parameters_dtr = [{'min_samples_leaf': [5, 10, 50, 100]}]
    clf_dtr = GridSearchCV(DecisionTreeRegressor(), tuned_parameters_dtr, scoring='neg_mean_absolute_error')
    models['dtr'] = clf_dtr

    # RandomForest
    models['rf'] = RandomForestRegressor()

    # SVR
    tuned_parameters_svm = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]}]
    clf_svm = GridSearchCV(SVR(), tuned_parameters_svm, scoring='neg_mean_absolute_error')
    models['svm'] = clf_svm

    models['bayes'] = BayesianRidge()
    models['knn'] = KNeighborsRegressor()

    # Neural Network
    models['nnet'] = MLPRegressor(activation='relu', hidden_layer_sizes=(25, 25, 25, 25), learning_rate='adaptive',
                                  learning_rate_init=0.2, max_iter=1000, solver='sgd', alpha=0.01, random_state=0,
                                  verbose=True)

    # models['nnet'] = MLPRegressor(activation='relu', hidden_layer_sizes=(25, 25, 25), learning_rate='adaptive',
    #                               learning_rate_init=0.1, early_stopping=False, momentum=0.99, max_iter=1000,
    #                               solver='sgd', alpha=0.01, random_state=0,
    #                               verbose=True)

    models['stacking'] = get_best_stacking()

    return models


# evaluate a given model using a train/test split
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)  # perform training
    y_pred = model.predict(X_test)
    score = mean_absolute_error(y_test, y_pred)
    return score, y_pred

def run_experiment(X,y):
    # perform train test split (train 80%, test 20%)
    X_train_prev, X_test_prev, y_train_prev, y_test_prev = train_test_split(X, y, test_size=0.2, random_state=1)

    # Data augmentation: generate samples within the interval defined by the errors

    # define number of samples to include
    n_samples = 10
    X_train, y_train = data_augmentation_with_uncertainties(X_train_prev, y_train_prev, n_samples)
    n_samples = 10  # sometimes we need more samples for the test
    X_test, y_test = data_augmentation_with_uncertainties(X_test_prev, y_test_prev, n_samples)

    # Normalize data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # get the models to evaluate
    models = get_models()

    # evaluate the models and store results
    results, names, predictions = list(), list(), list()
    for name, model in models.items():
        scores, y_pred = evaluate_model(model, X_train_norm, y_train, X_test_norm, y_test)
        results.append(scores)
        predictions.append(y_pred)
        names.append(name)
        print('>%s %.3f' % (name, mean(scores)))

    return results, names

##############################
# Mass and radius estimation #
##############################

# Read raw data
data = get_dataset()

# Chose the target variable to estimate (mass 'M' or radius 'R')
target = 'M'
# target = 'R'

# to read target with errors eM1/eM2 or eR1/eR2
y_ser = data.loc[:, [target, 'e' + target + '1', 'e' + target + '2']]
y = y_ser.to_numpy()

# Declare the combinations of features to use
experiments = dict()
experiments['exp1'] = ['Teff', 'eTeff1', 'eTeff2', 'L', 'eL1', 'eL2',]
experiments['exp2'] = ['Teff', 'eTeff1', 'eTeff2', 'L', 'eL1', 'eL2', 'logg', 'elogg1', 'elogg2']
experiments['exp3'] = ['Teff', 'eTeff1', 'eTeff2', 'L', 'eL1', 'eL2', 'logg', 'elogg1', 'elogg2', 'Meta', 'eMeta1', 'eMeta2']
experiments['exp4'] = ['Teff', 'eTeff1', 'eTeff2', 'L', 'eL1', 'eL2', 'Meta', 'eMeta1', 'eMeta2']
# Should we include more options

num_experiments = len(experiments)

fig, ax = pyplot.subplots()
for e, exp_conf in experiments.items():
    X_ser = data.loc[:,exp_conf]
    # Selection of the data to be used for the regression (read also the errors)
    X = X_ser.to_numpy()
    print('Running experiment >', e)
    results, model_names = run_experiment(X, y)
    num_models = len(model_names)
    #show results
    ax.plot(np.arange(num_models),results, label=e)


#Show all experiments
ax.legend()
ax.grid(True)
ax.set_xticklabels((model_names))
pyplot.ylabel('MAE')
pyplot.show()


