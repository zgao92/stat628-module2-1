"""Contains the functions which perform all data processing and NLP tasks.

The data processing is split up into separate functions for a few reasons:

  - To avoid duplicate computation (e.g., if you want to redo the NLP, you
    shouldn't have to start over with the raw JSON files)
  - To minimize the amount of RAM necessary to construct the datasets
  - To keep the code tidy

You can run this file from the command line or simply call `check_files()` to
create all of the cleaned and formatted data files. If the files already exist,
they won't be re-created.
"""

import ast
import csv
import json
import os
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd
import scipy as sp

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def map_to_dict(x):
    """Evaulates dict strings and maps everything else to an empty dict."""
    result = ast.literal_eval(str(x))  # Like `eval()` but safer
    return result if isinstance(result, dict) else {}


def expand_dict_column(column):
    """Expands a column of dicts into multiple individual columns.

    Args:
        column: Pandas Series of dict values.

    Returns:
        DataFrame with the dicts split up into individual columns.
    """
    tqdm.pandas()
    col_name = column.name
    column_as_dicts = column.progress_apply(map_to_dict).to_list()
    expanded = pd.DataFrame.from_records(column_as_dicts)
    expanded.columns = ['{}.{}'.format(col_name, x) for x in expanded.columns]
    return expanded


def json_to_dataframe(json_path):
    """Converts a file containing rows of JSON objects to a DataFrame.

    Args:
        json_path: Filepath of the JSON data.

    Returns:
        DataFrame containing the keys of the JSON objects as features.
    """
    with open(json_path, mode='r') as f:
        content = [json.loads(line) for line in tqdm(f)]
    return pd.DataFrame(content)


def create_indicators(column, na_values=[]):
    """Expands `column` into indicator columns and removes NAs.

    Args:
        column: Pandas Series.
        na_values: Manually specify additional values to treat as NaN/missing.

    Returns:
        DataFrame of indicator columns.
    """
    na_values = [np.nan, "'none'", 'None', "u'none'"] + na_values
    indicators = pd.DataFrame(index=column.index)
    for level in set(column.values) - set(na_values):
        feature_name = '{}.{}'.format(column.name,
                                      ''.join(re.findall('\\w', str(level))))
        indicators[feature_name] = (column == level).astype(np.uint8)
    return indicators


def make_csv_files():
    """Reads the raw JSON files and reformats/saves them as CSV files."""
    business_columns_to_skip = ['city', 'hours', 'name', 'postal_code']
    review_columns_to_skip = ['date']
    train_state_features = set()

    for dataset in ['train', 'test']:
        print('Transforming raw {} data...'.format(dataset))

        # Convert JSONs -> DataFrames and merge business data with review data
        business = json_to_dataframe('data/business_{}.json'.format(dataset))
        business.drop(business_columns_to_skip, axis=1, inplace=True)
        review = json_to_dataframe('data/review_{}.json'.format(dataset))
        review.drop(review_columns_to_skip, axis=1, inplace=True)
        df = pd.merge(review, business, how='left', on='business_id')

        # Collect all of the unique categories in the dataset
        print('Extracting categories in the {} data...'.format(dataset))
        found_categories = [x.split(',') if isinstance(x, str) else []
                            for x in df['categories'].to_list()]

        # Strip leading/trailing spaces from these categories
        found_categories = [[c.strip() for c in l] for l in found_categories]

        # Create indicator columns for the top 25 categories
        categories_to_keep = [
            'Restaurants', 'Food', 'Nightlife', 'Bars',
            'American (Traditional)', 'American (New)', 'Breakfast & Brunch',
            'Event Planning & Services', 'Shopping', 'Sandwiches',
            'Beauty & Spas', 'Arts & Entertainment', 'Mexican', 'Burgers',
            'Pizza', 'Italian', 'Hotels & Travel', 'Seafood', 'Coffee & Tea',
            'Japanese', 'Home Services', 'Desserts', 'Automotive', 'Chinese',
            'Sushi Bars'
        ]
        category_feature_names = []
        for category in tqdm(categories_to_keep):
            feature_name = 'category.{}'.format(category.replace(' ', ''))
            category_feature_names.append(feature_name)
            df[feature_name] = [1 if category in sublist else 0
                                for sublist in found_categories]

        # Create an "Other" column for the rest of the categories
        df['category.Other'] = 1 - np.clip(
            df[category_feature_names].sum(axis=1), 0, 1)

        # Create indicator variables for state variables
        states = create_indicators(df['state'])
        df = df.join(states).drop(['state'], axis=1)

        if dataset == 'train':
            train_state_features.update(states.columns)
        else:
            # Remove {test states} \ {train states}
            df.drop(list(set(states.columns).difference(train_state_features)),
                    axis=1, inplace=True)

            # Add {train states} \ {test states}
            for feature in train_state_features.difference(states.columns):
                df[feature] = np.zeros(df.shape[0], np.int64)

        # Drop columns that are not needed any more
        df.drop(['business_id', 'categories'], axis=1, inplace=True)

        # Fill some missing values
        df['text'].fillna('', inplace=True)
        df['attributes'].fillna('{}', inplace=True)
        df['longitude_missing'] = df['longitude'].isna().astype(np.uint8)
        df['latitude_missing'] = df['latitude'].isna().astype(np.uint8)
        df['longitude'].fillna(df['longitude'].mean(), inplace=True)
        df['latitude'].fillna(df['latitude'].mean(), inplace=True)

        # Save to CSV
        df.to_csv('data/{}.csv'.format(dataset),
                  index=False, quoting=csv.QUOTE_ALL)

    # Resave the hotel data as a separate CSV file
    train = pd.read_csv('data/train.csv')
    train.loc[train['category.Hotels&Travel'] == 1].to_csv(
        'data/hotel.csv', index=False, quoting=csv.QUOTE_ALL)


def extract_attributes():
    """Extracts the attributes from the cleaned CSV files and resaves them."""
    to_skip = [
        'attributes.Ambience',
        'attributes.BestNights',
        'attributes.BusinessParking',
        'attributes.DietaryRestrictions',
        'attributes.GoodForMeal',
        'attributes.HairSpecializesIn',
        'attributes.Music',
    ]
    for dataset in ['full', 'hotel']:
        # Read the attributes as a single Pandas Series
        if dataset == 'full':
            print('Expanding attributes (takes > 30 minutes)...')
            train = pd.read_csv('data/train.csv', usecols=['attributes'])
            test = pd.read_csv('data/test.csv', usecols=['attributes'])

            # Fill non-overlapping columns with 0
            missing_test_columns = set(train.columns).difference(test.columns)
            test[list(missing_test_columns)] = \
                np.zeros((test.shape[0], len(missing_test_columns)), np.int64)
            for feature in set(train.columns).difference(test.columns):
                test[feature] = np.zeros((test.shape[0],), np.int64)
            test[list(set(train.columns).difference(test.columns))] = 0

            column = pd.concat([train, test], ignore_index=True)
        elif dataset == 'hotel':
            print('Expanding attributes for hotel dataset...')
            column = pd.read_csv('data/hotel.csv', usecols=['attributes'])
        column = column.iloc[:, 0]

        # Expand the dict attributes into separate columns
        attributes = expand_dict_column(column)
        attributes.drop(to_skip, axis=1, inplace=True, errors='ignore')

        # Go through all the attribute columns
        columns_to_expand = attributes.columns.to_list()
        for column in tqdm(columns_to_expand):
            attributes = attributes.join(
                create_indicators(attributes[column])).drop(column, axis=1)

        # Fix a problem in the data where some columns are duplicated
        columns_to_check = attributes.columns.to_list()
        for feature in columns_to_check:
            problematic_feature = '{}.{}.u{}'.format(*feature.split('.'))
            if problematic_feature in attributes.columns:
                attributes[feature] += attributes[problematic_feature]
                attributes.drop(problematic_feature, axis=1, inplace=True)

        # Save the attribute names and attribute matrix
        with open('data/attribute_names_{}.pickle'.format(dataset), 'wb') as f:
            pickle.dump(attributes.columns.to_list(), f)

        X = sp.sparse.csr_matrix(attributes.values)
        with open('data/attributes_{}.pickle'.format(dataset), 'wb') as f:
            pickle.dump(X, f)


def perform_nlp():
    """Performs the NLP steps and saves the matrix and feature names to files.

    This function creates two files: 'data/text.pickle' and
    'data/vocabulary.pickle'. They contain the matrix of word-features and the
    list of column names for those word-features, respectively.

    The reason the NLP is done here is so that the rest of the dataset isn't
    loaded in memory at the same time, since this is one of the most
    computationally intensive steps of the project.
    """
    for dataset in ['full', 'hotel']:
        # Read the text as a single array
        if dataset == 'full':
            print('Performing NLP (could take >1 hour!)...')
            train = pd.read_csv('data/train.csv', usecols=['text'])
            test = pd.read_csv('data/test.csv', usecols=['text'])
            column = pd.concat([train, test], ignore_index=True)
        elif dataset == 'hotel':
            print('Performing NLP on hotel data...')
            column = pd.read_csv('data/hotel.csv', usecols=['text'])
        text = column.iloc[:, 0].fillna('').values

        # Get the English stopwords included with NLTK
        try:
            stop_words = stopwords.words('english')
        except LookupError:
            import nltk
            nltk.download('stopwords')
            stop_words = stopwords.words('english')

        # Custom tokenizer function
        stemmer = SnowballStemmer('english', ignore_stopwords=True)
        regex_tokenizer = RegexpTokenizer(r'[a-z\']+')

        def tokenizer(x):
            """Finds all words in the text and normalizes the word tenses.

            For example, it would map the following words as follows:
                cook -> cook
                cooking -> cook
                cooked -> cook
            """
            return [stemmer.stem(w)
                    for w in regex_tokenizer.tokenize(x.lower())]

        # Fit the TF/IDF vectorizer to the review text data
        tfidf = TfidfVectorizer(strip_accents='ascii',
                                analyzer='word',
                                stop_words=stop_words,
                                tokenizer=tokenizer,
                                max_features=10000)

        start = time.time()
        text = tfidf.fit_transform(text)
        end = time.time()
        print('Completed TF/IDF vectorization in {} minutes.'.format(
            (end - start) / 60))

        # Save these objects for later use
        with open('data/text_{}.pickle'.format(dataset), 'wb') as f:
            pickle.dump(text, f, pickle.HIGHEST_PROTOCOL)
        with open('data/vocabulary_{}.pickle'.format(dataset), 'wb') as f:
            pickle.dump(tfidf.get_feature_names(), f, pickle.HIGHEST_PROTOCOL)


def make_sparse_matrix():
    """Creates the final sparse matrix to feed to the models.

    The matrix is a sparse matrix with shape (n_train + n_test, p) and will be
    saved to a file `data/sparse.npz`. To load it, run:
    ```
    X = sp.sparse.load_npz('data/sparse.npz')
    ```
    The list of feature names are saved to `data/feature_names.pickle`.
    """
    for dataset in ['full', 'hotel']:
        # Read the CSV file(s) and drop the columns train and test don't share
        if dataset == 'full':
            train = pd.read_csv('data/train.csv').drop(
                ['text', 'stars', 'attributes'], axis=1)
            test = pd.read_csv('data/test.csv').drop(
                ['text', 'KaggleID', 'attributes'], axis=1)

            df = pd.concat([train, test], ignore_index=True, sort=True)
        elif dataset == 'hotel':
            df = pd.read_csv('data/hotel.csv').drop(
                ['text', 'stars', 'attributes'], axis=1)

        # Get the feature names and save them before continuing
        with open('data/attribute_names_{}.pickle'.format(dataset), 'rb') as f:
            attribute_names = pickle.load(f)
        with open('data/vocabulary_{}.pickle'.format(dataset), 'rb') as f:
            vocabulary = pickle.load(f)
        feature_names = df.columns.to_list() + attribute_names + vocabulary
        with open('data/feature_names_{}.pickle'.format(dataset), 'wb') as f:
            pickle.dump(feature_names, f)

        # Join the matrices into a single feature set
        with open('data/text_{}.pickle'.format(dataset), 'rb') as f:
            text = pickle.load(f)
        with open('data/attributes_{}.pickle'.format(dataset), 'rb') as f:
            attributes = pickle.load(f)

        X = sp.sparse.hstack(
            (sp.sparse.csr_matrix(df.values), attributes, text))

        # Save sparse dataset to files
        sp.sparse.save_npz('data/sparse_{}.npz'.format(dataset), X)


def check_files():
    """Checks to make sure all of the files have been created.

    If a file has not been created or is missing, it is recreated."""
    try:
        os.chdir(os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir)))
    except NameError:
        if not os.path.exists('data/'):
            print('You are running this in the wrong directory! You should run'
                  'this in the project root where the `data/` directory is.')
            exit(1)

    # Check all files and create them if they are missing
    if not all(os.path.exists(p) for p in ['data/business_train.json',
                                           'data/business_test.json',
                                           'data/review_train.json',
                                           'data/review_test.json']):
        print('The data is missing! Put the raw JSON files in `data/`.')
        sys.exit(1)
    if not all(os.path.exists(p) for p in ['data/train.csv',
                                           'data/test.csv',
                                           'data/hotel.csv']):
        print('The cleaned data needs to be created...')
        make_csv_files()
    if not all(os.path.exists(p) for p in ['data/attribute_names_full.pickle',
                                           'data/attribute_names_hotel.pickle',
                                           'data/attributes_full.pickle',
                                           'data/attributes_hotel.pickle']):
        print('The attributes need to be extracted...')
        extract_attributes()
    if not all(os.path.exists(p) for p in ['data/text_full.pickle',
                                           'data/vocabulary_full.pickle',
                                           'data/text_hotel.pickle',
                                           'data/vocabulary_hotel.pickle']):
        print('The NLP data needs to be created...')
        perform_nlp()
    if not all(os.path.exists(p) for p in ['data/sparse_full.npz',
                                           'data/feature_names_full.pickle',
                                           'data/sparse_hotel.npz',
                                           'data/feature_names_hotel.pickle']):
        print('The sparse data matrix needs to be created...')
        make_sparse_matrix()
    print('Done! All the files exist.')


if __name__ == '__main__':
    check_files()
