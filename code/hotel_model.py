"""Contains the random forest model used to analyze the hotel data.
"""

import csv
import pickle

import numpy as np
import scipy as sp
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

try:
    from functions import check_files
except ImportError:
    from code.functions import check_files

# Make sure the data has been prepared
check_files()

# Read the sparse data
print('Loading the hotel data...')
X = sp.sparse.load_npz('data/sparse_hotel.npz').tocsr()
y = pd.read_csv('data/hotel.csv', usecols=['stars']).iloc[:, 0].values
with open('data/feature_names_hotel.pickle', 'rb') as f:
    feature_names = pickle.load(f)

# Random forest model
clf = RandomForestRegressor(n_estimators=500,
                            max_depth=10,
                            max_features='auto',
                            oob_score=True,
                            n_jobs=-1,
                            random_state=1,
                            verbose=2,
                            warm_start=True)
clf.fit(X, y)

# Unbiased estimator of the forest's prediction accuracy
print(clf.oob_score_)

# Save fitted model
with open('data/model_hotel.pickle', 'wb') as f:
    pickle.dump(clf, f)

# Obtain the importance of each feature name and sort them by importance
importances = clf.feature_importances_ * 1000.
feature_importances = [(f, i) for f, i in zip(feature_names, importances)]
feature_importances.sort(key=lambda x: x[1], reverse=True)
importances_table = pd.DataFrame(np.array(feature_importances),
                                 columns=['Feature Name', 'Importance'])
importances_table.to_csv('data/feature_importances_hotel.csv', index=False)

# Resave the hotel review text alongside the top 1000 most important features
# This makes it easier to dig into the meaning of words and phrases in the text
hotel_text = pd.read_csv('data/hotel.csv', usecols=['text']).values
top_1000_features = [f[0] for f in feature_importances[:1000]]
top_1000_indices = [i for i, e in enumerate(feature_names)
                    if e in top_1000_features]
hotel = np.hstack([y.reshape(-1, 1),
                   hotel_text,
                   X[:, top_1000_indices].toarray()])
hotel = pd.DataFrame(hotel, columns=['stars', 'text'] + top_1000_features)
hotel.to_csv('data/hotel_top_1000_analysis.csv',
             index=False, quoting=csv.QUOTE_ALL)
