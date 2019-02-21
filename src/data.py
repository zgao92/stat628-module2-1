"""Contains the functions which load, clean, and structure the dataset.
"""

import ast
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

# The most frequent 25 columns ordered by # occurrences
CATEGORIES_TO_KEEP = [
    'Restaurants', 'Food', 'Nightlife', 'Bars', 'American (Traditional)',
    'American (New)', 'Breakfast & Brunch', 'Event Planning & Services',
    'Shopping', 'Sandwiches', 'Beauty & Spas', 'Arts & Entertainment',
    'Mexican', 'Burgers', 'Pizza', 'Italian', 'Hotels & Travel', 'Seafood',
    'Coffee & Tea', 'Japanese', 'Home Services', 'Desserts', 'Automotive',
    'Chinese', 'Sushi Bars'
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
    temp_df = pd.DataFrame(df[feature].apply(map_to_dict).to_list(),
                           dtype=object)
    temp_df = temp_df.add_prefix('{}.'.format(feature))
    return df.drop(feature, axis=1).join(temp_df)


def transform_raw_files():
    """Reads the raw JSON files and reformats them into DataFrames.

    Returns:
        train, test DataFrames.
    """
    # Read in the raw JSON files and merge them
    print('Reading training data...')
    business_train = json_to_dataframe('data/business_train.json')
    review_train = json_to_dataframe('data/review_train.json')

    print('Merging the business and review training data...')
    train = pd.merge(review_train, business_train, how='left',
                     on='business_id')

    train = clean_file(train)

    print('Saving the clean training data to CSV...')
    train.to_csv('data/train.csv', index=False)

    print('Reading test data...')
    business_test = json_to_dataframe('data/business_test.json')
    review_test = json_to_dataframe('data/review_test.json')

    print('Merging the business and review test data...')
    test = pd.merge(review_test, business_test, how='left', on='business_id')

    test = clean_file(test)

    print('Saving the clean test data to CSV...')
    test.to_csv('data/test.csv', index=False)


def clean_file(df):
    """Performs cleaning on the dataset (expands columns, removes NAs, etc.).

    Args:
        df: Either the train or test DataFrame object.

    Returns:
        Cleaned DataFrame.
    """
    # There are 1301 unique categories, I chose to keep only the top 25
    print('Expanding categories...')

    # Split each row of 'categories' into a list of categories
    found_categories = [x.split(',') if isinstance(x, str) else []
                        for x in df['categories'].to_list()]

    # Strip leading/trailing spaces from these categories
    found_categories = [[c.strip() for c in l] for l in found_categories]

    # Create a DataFrame of indicator columns for the top categories
    print('Converting categories to indicator variables...')
    categories = pd.DataFrame()
    for category in tqdm(CATEGORIES_TO_KEEP):
        feature_name = 'category.{}'.format(category.replace(' ', ''))
        categories[feature_name] = [1 if category in sublist else 0
                                    for sublist in found_categories]

    # Create an "Other" column for the rest of the categories
    categories['category.Other'] = 1 - np.clip(categories.sum(axis=1), 0, 1)

    # Merge the categorical indicator features with the main DataFrame
    df = df.drop('categories', axis=1).join(categories)

    # Drop attributes for now since it takes a long time to read
    df.drop('attributes', axis=1, inplace=True)

    return df


def read_files(nrows=None):
    """Gets the cleaned train, test datasets from the CSV or JSON files.

    If the CSV files containing the cleaned data don't exist yet, they are
    created after reading and transforming the data in the raw JSON files.

    Returns:
        train, test DataFrames.
    """
    # Create the clean dataset if it hasn't been done already
    if not all(os.path.exists(p) for p in ['data/train.csv', 'data/test.csv']):
        transform_raw_files()

    # Read the clean data files
    print('Reading CSV files...')
    train = pd.read_csv('data/train.csv', nrows=nrows, dtype=object)
    test = pd.read_csv('data/test.csv', nrows=nrows, dtype=object)

    print('Done!')

    return train, test
