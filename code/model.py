"""Runs the model, generates feature importances, and makes predictions."""

import pickle

import numpy as np
import scipy as sp
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split

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

# Split the data into training and validation sets
train_i, valid_i = train_test_split(train_indices,
                                    test_size=0.2,
                                    random_state=1)
dtrain_train = dtrain.slice(train_i)
dtrain_valid = dtrain.slice(valid_i)

# Boosted trees model
general_params = {
    'booster': 'gbtree',
    'verbosity': 2,
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
bst = xgb.train(params={**general_params, **booster_params},
                dtrain=dtrain_train,
                num_boost_round=1000,
                evals=[(dtrain_valid, 'valid')],
                early_stopping_rounds=10)

# Save fitted model
with open('data/model.pickle', 'wb') as f:
    pickle.dump(bst, f)


def get_feature_importance(booster=bst):
    """Returns a sorted dictionary of {word: importance} key-value pairs."""
    importance = [[k, v] for k, v in booster.get_score(importance_type='gain')]
    importance = sorted(importance, key=lambda x: x[1], reverse=True)
    return importance


# Save feature importances
with open('data/feature_importances.pickle', 'wb') as f:
    pickle.dump(get_feature_importance(bst), f)

# Make predictions and write to file
y_test = bst.predict(dtest)
final = pd.DataFrame({'ID': np.arange(1, test_indices.shape[0] + 1),
                      'Expected': y_test})
final.to_csv('data/predictions.csv', index=False)
