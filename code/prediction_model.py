"""Contains the XGBoost model used to make predictions for the full dataset.
"""

import pickle

import numpy as np
import scipy as sp
import pandas as pd
import xgboost as xgb

try:
    from functions import check_files
except ImportError:
    from code.functions import check_files

# Make sure the data has been prepared
check_files()


# Read the sparse data in as train and test
print('Loading the data...')
X = sp.sparse.load_npz('data/sparse_full.npz').tocsr()
y = pd.read_csv('data/train.csv', usecols=['stars']).iloc[:, 0].values
with open('data/feature_names_full.pickle', 'rb') as f:
    feature_names = pickle.load(f)

train_indices = np.arange(0, 5364626)
test_indices = np.arange(5364626, 6685900)
dtrain = xgb.DMatrix(X[train_indices, :], y, feature_names=feature_names)
dtest = xgb.DMatrix(X[test_indices, :], feature_names=feature_names)

# Boosted trees model
print('Training the model...')
general_params = {
    'booster': 'gbtree',
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
    'subsample': 0.5,  # default=1.; proportion of points to sample each round
    'lambda': 1,  # default=1, L2 regularization
    'alpha': 0,  # default=0, L1 regularization
}
bst = xgb.train(params={**general_params, **booster_params},
                dtrain=dtrain,
                num_boost_round=2000,
                evals=[(dtrain, 'train')])

# Save fitted model
with open('data/model_full.pickle', 'wb') as f:
    pickle.dump(bst, f)

# Make predictions and write to file
y_test = bst.predict(dtest)
final = pd.DataFrame({'ID': np.arange(1, test_indices.shape[0] + 1),
                      'Expected': y_test})
final.to_csv('data/predictions.csv', index=False)
