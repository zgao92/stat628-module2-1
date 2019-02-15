"""Contains the functions which load, clean, and structure the dataset.
"""

import os
import json

import pandas as pd
from tqdm import tqdm

FILEPATHS = {
    'business_train': ('data/business_train.json', 'data/business_train.csv'),
    'business_test': ('data/business_test.json', 'data/business_test.csv'),
    'review_train': ('data/review_train.json', 'data/review_train.csv'),
    'review_test': ('data/review_test.json', 'data/review_test.csv')
}


def json_to_csv(json_path, csv_path):
    """Converts a file containing JSON objects to a tabular CSV file.

    Args:
        json_path: Filepath of the JSON data.
        csv_path: Filepath of the CSV file to be created.
    """
    with open(json_path, mode='r') as f:
        content = [json.loads(line) for line in tqdm(f)]
    pd.DataFrame(content).to_csv(csv_path, index=False)


def read_files(nrows=None):
    """Reads all four files and returns them as DataFrames.

    If the CSV files exists, the function reads that file. Otherwise, it reads
    the raw JSON lines and converts it to a CSV.

    Args:
        nrows: Read at most this many lines of each file or all rows if None.

    Returns:
        business_train, business_test, review_train, review_test DataFrames.
    """
    datasets = []
    for dataset, (json_path, csv_path) in FILEPATHS.items():
        if not os.path.exists(csv_path):
            print('Converting {}.json to CSV...'.format(dataset))
            json_to_csv(json_path, csv_path)
        print('Reading file: {}.csv'.format(dataset))
        datasets.append(pd.read_csv(csv_path, nrows=nrows))
    return datasets
