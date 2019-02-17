"""The primary Python src used for modeling, prediction, and analysis.
"""

import os

from src.data import read_files

# Set the working directory to the project root folder `stat628-module2`
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# Load the datasets
# Set nrows=X to read max X rows from each file, or nrows=None to read all rows
train, test = read_files(nrows=100000)

# TODO: everything else
#   - clean the data / feature engineering
#   - run the models
#   - generate predictions
#   - assess accuracy / save predictions to file
#   - analysis
