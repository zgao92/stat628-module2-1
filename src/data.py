"""Contains the functions which load, clean, and structure the dataset.
"""

import ast
import json
import os
import warnings

import pandas as pd
from tqdm import tqdm

RAW_FILEPATHS = {
    'business_train': 'data/business_train.json',
    'business_test': 'data/business_test.json',
    'review_train': 'data/review_train.json',
    'review_test': 'data/review_test.json',
}

CLEAN_FILEPATHS = {
    'train': 'data/train.h5',
    'test': 'data/test.h5'
}

DICT_FEATURES = [
    # These take a long time to expand, so only select the ones we really want
    'attributes',
    # 'hours',
    # 'attributes.Ambience',
    # 'attributes.BusinessParking',
    # 'attributes.GoodForMeal',
]


def json_to_dataframe(json_path):
    """Converts a file containing JSON objects to a DataFrame.

    Args:
        json_path: Filepath of the JSON data.

    Returns:
        DataFrame containing the keys of the JSON objects as features.
    """
    with open(json_path, mode='r') as f:
        content = [json.loads(line) for line in tqdm(f)]
    return pd.DataFrame(content)


def map_to_dict(x):
    """Evaulates dict strings and maps everything else to an empty dict."""
    result = ast.literal_eval(str(x))  # Like `eval()` but safer
    return result if isinstance(result, dict) else {}


def expand_dict_feature(df, feature):
    """Expands a column of dicts in `df` into multiple individual columns.

    Args:
        df: DataFrame object containing the column to be expanded.
        feature: Name of the column to expand.

    Returns:
        DataFrame with `feature` split up into individual columns.
    """
    df.loc[df[feature].isna(), feature] = '{}'
    temp_df = pd.DataFrame(df[feature].apply(map_to_dict).to_list())
    temp_df = temp_df.add_prefix('{}.'.format(feature))
    return df.drop(feature, axis=1).join(temp_df)


def transform_raw_files():
    """Reads the raw JSON files and reformats them into DataFrames.
    
    Returns:
        train, test DataFrames.
    """

    # Read in each raw JSON file
    datasets = {}
    for dataset, path in RAW_FILEPATHS.items():
        print('Reading {}.json as a DataFrame...'.format(dataset))
        datasets[dataset] = json_to_dataframe(path)

    # Merge the review and business datasets
    print('Merging business and review data...')
    train = pd.merge(datasets['review_train'], datasets['business_train'],
                     how='left', on='business_id')
    test = pd.merge(datasets['review_test'], datasets['business_test'],
                    how='left', on='business_id')

    # Convert columns containing dicts to individual columns
    print('Expanding dict columns (this takes a while)...')
    for feature in tqdm(DICT_FEATURES):
        test = expand_dict_feature(test, feature)
        train = expand_dict_feature(train, feature)

    return train, test


def read_files(nrows=None):
    """Gets the cleaned train, test datasets from the HDF5 or JSON files.

    HDF5 is a high-performance method of storing tabular data (much faster than
    CSV). If the HDF5 files containing the cleaned data don't exist yet, they
    are created after reading and transforming the data in the raw JSON files.

    Returns:
        train, test DataFrames.
    """
    # If the cleaned HDF5 files have already been created, use those
    if all(os.path.exists(path) for path in CLEAN_FILEPATHS.values()):
        train = pd.read_hdf(CLEAN_FILEPATHS['train'], key='train', stop=nrows)
        test = pd.read_hdf(CLEAN_FILEPATHS['test'], key='test', stop=nrows)

    # Otherwise, create them from the raw JSON files
    else:
        train, test = transform_raw_files()
        warnings.simplefilter(
            action='ignore', category=pd.errors.PerformanceWarning)
        train.to_hdf(CLEAN_FILEPATHS['train'], key='train')
        test.to_hdf(CLEAN_FILEPATHS['test'], key='train')

        if nrows is not None:
            train = train.head(nrows)
            test = test.head(nrows)

    return train, test
