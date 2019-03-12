"""Searches for best XGBoost hyperparameters."""

import pickle

import numpy as np
import scipy as sp
import pandas as pd
import xgboost as xgb

from scipy.stats.distributions import expon, randint, uniform
from sklearn.model_selection import ParameterSampler, train_test_split
from tqdm import tqdm

try:
    from functions import check_files
except ImportError:
    from code.functions import check_files

# Make sure the data has been prepared
check_files()

# Read the sparse data in as train and test
print('Loading the data...')
X = sp.sparse.load_npz('data/sparse.npz').tocsr()
y = pd.read_csv('data/train.csv', usecols=['stars']).iloc[:, 0].values
with open('data/feature_names.pickle', 'rb') as f:
    feature_names = pickle.load(f)

train_indices = np.arange(0, 5364626)
test_indices = np.arange(5364626, 6685900)
dtrain = xgb.DMatrix(X[train_indices, :], y, feature_names=feature_names)
dtest = xgb.DMatrix(X[test_indices, :], feature_names=feature_names)
del X, y

# Keep track of our default parameters
general_params = {
    'booster': 'gbtree',
    'verbosity': 0,
    'tree_method': 'approx',
    'max_bin': 128,
    'eval_metric': 'rmse',
    'seed': 1,
}
booster_params = {
    'eta': 0.1,  # default=0.3
    'gamma': 0.,  # default=0.; larger => more conservative
    'max_depth': 6,  # default=6
    'min_child_weight': 1,  # default=1; larger => more conservative
    'subsample': 1.,  # default=1.; proportion of points to sample each round
    'lambda': 1,  # default=1, L2 regularization
    'alpha': 0,  # default=0, L1 regularization
}

# Parameter space to search over
param_dist = {
    'eta': [0.3],
    'gamma': expon(),
    'max_depth': randint(3, 10),
    'min_child_weight': randint(1, 10),
    'subsample': uniform(0.5, 0.5),
    'lambda': expon(),
    'alpha': expon()
}
sampler = ParameterSampler(param_dist, n_iter=30, random_state=1)

# Perform the search
best_score = np.Inf
best_params = {**general_params, **booster_params}

# Repeatedly sample parameters from the above distributions
print('Testing hyperparameters...')
for point in tqdm(sampler):
    current_params = best_params.copy()
    current_params.update(point)

    # Split the data into training and validation sets
    train_i, valid_i = train_test_split(train_indices,
                                        test_size=0.2,
                                        random_state=1)
    dtrain_train = dtrain.slice(train_i)
    dtrain_valid = dtrain.slice(valid_i)

    # Train the model using the given parameters
    bst = xgb.train(params=current_params,
                    dtrain=dtrain_train,
                    num_boost_round=20,
                    evals=[(dtrain_valid, 'valid')],
                    early_stopping_rounds=10)

    # Keep track of the best parameters and their corresponding score
    current_score = np.float(bst.attributes()['best_score'])
    if current_score < best_score:
        best_score = current_score
        best_params = current_params

print('Best score: {}'.format(best_score))

# Save the best parameters
with open('best_parameters.pickle', 'wb') as f:
    pickle.dump(best_params, f)
