"""The primary Python code used for modeling, prediction, and analysis.
"""

import os

from code.data import read_files

# Set the working directory to the project root folder `stat628-module2`
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# Load the files; set `nrows=X` to read only X rows of each file
bus_train, bus_test, rev_train, rev_test = read_files(nrows=100000)

print(bus_train.shape)
print(bus_test.shape)
print(rev_train.shape)
print(rev_test.shape)

print(bus_test.head())
print(rev_test.head())

# TODO: everything else
#   - clean the data / feature engineering
#   - run the models
#   - generate predictions
#   - assess accuracy / save predictions to file
#   - analysis
