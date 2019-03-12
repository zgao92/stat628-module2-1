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


def expand_categories(column):
    """One-hot-encodes the top 25 categories and labels the rest as "Other".

    Args:
        column: Pandas Series containing lists of categories.

    Returns:
        DataFrame of expanded categories.
    """
    categories_to_keep = [
        'Restaurants', 'Food', 'Nightlife', 'Bars', 'American (Traditional)',
        'American (New)', 'Breakfast & Brunch', 'Event Planning & Services',
        'Shopping', 'Sandwiches', 'Beauty & Spas', 'Arts & Entertainment',
        'Mexican', 'Burgers', 'Pizza', 'Italian', 'Hotels & Travel', 'Seafood',
        'Coffee & Tea', 'Japanese', 'Home Services', 'Desserts', 'Automotive',
        'Chinese', 'Sushi Bars'
    ]

    # Split each row of 'categories' into a list of categories
    found_categories = [x.split(',') if isinstance(x, str) else []
                        for x in column.to_list()]

    # Strip leading/trailing spaces from these categories
    found_categories = [[c.strip() for c in l] for l in found_categories]

    # Create a DataFrame of indicator columns for the top categories
    categories = pd.DataFrame()
    for category in tqdm(categories_to_keep):
        feature_name = 'category.{}'.format(category.replace(' ', ''))
        categories[feature_name] = [1 if category in sublist else 0
                                    for sublist in found_categories]

    # Create an "Other" column for the rest of the categories
    categories['category.Other'] = 1 - np.clip(categories.sum(axis=1), 0, 1)

    return categories


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

    print('Transforming raw train data...')
    business_train = json_to_dataframe('data/business_train.json').drop(
        business_columns_to_skip, axis=1)
    review_train = json_to_dataframe('data/review_train.json').drop(
        review_columns_to_skip, axis=1)

    train = pd.merge(review_train, business_train,
                     how='left', on='business_id')  # train => (5364626, 8)
    train = train.join(expand_categories(train['categories'])).drop(
        'categories', axis=1)
    train['attributes'].fillna('{}', inplace=True)

    train.to_csv('data/train.csv', index=False, quoting=csv.QUOTE_ALL)

    print('Transforming raw test data...')
    business_test = json_to_dataframe('data/business_test.json').drop(
        business_columns_to_skip, axis=1)
    review_test = json_to_dataframe('data/review_test.json').drop(
        review_columns_to_skip, axis=1)

    test = pd.merge(review_test, business_test,
                    how='left', on='business_id')  # test  => (1321274, 8)
    test = test.join(expand_categories(test['categories'])).drop(
        'categories', axis=1)
    test['attributes'].fillna('{}', inplace=True)

    test.to_csv('data/test.csv', index=False, quoting=csv.QUOTE_ALL)


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

    # Read the cleaned CSV files and concatenate them into a single DataFrame
    train = pd.read_csv('data/train.csv', usecols=['attributes'])
    test = pd.read_csv('data/test.csv', usecols=['attributes'])
    column = pd.concat([train, test], ignore_index=True).iloc[:, 0]

    # Expand the dict attributes into separate columns
    print('Expanding attributes (takes >30 minutes)...')
    attributes = expand_dict_column(column)
    attributes.drop(to_skip, axis=1, inplace=True)

    # Go through all the attribute columns
    print('Converting categories to indicators...')
    columns_to_expand = attributes.columns.to_list()
    for column in tqdm(columns_to_expand):
        attributes = attributes.join(
            create_indicators(attributes[column])).drop(column, axis=1)

    # Manually fix some of the remaining quirks
    print('Cleaning up...')
    attributes['attributes.Alcohol.beer_and_wine'] += \
        attributes['attributes.Alcohol.ubeer_and_wine']
    attributes['attributes.Alcohol.full_bar'] += \
        attributes['attributes.Alcohol.ufull_bar']
    attributes['attributes.BYOBCorkage.yes_corkage'] += \
        attributes['attributes.BYOBCorkage.uyes_corkage']
    attributes['attributes.NoiseLevel.quiet'] += \
        attributes['attributes.NoiseLevel.uquiet']
    attributes['attributes.NoiseLevel.average'] += \
        attributes['attributes.NoiseLevel.uaverage']
    attributes['attributes.NoiseLevel.loud'] +=\
        attributes['attributes.NoiseLevel.uloud']
    attributes['attributes.NoiseLevel.very_loud'] += \
        attributes['attributes.NoiseLevel.uvery_loud']
    attributes['attributes.RestaurantsAttire.casual'] += \
        attributes['attributes.RestaurantsAttire.ucasual']
    attributes['attributes.RestaurantsAttire.dressy'] += \
        attributes['attributes.RestaurantsAttire.udressy']
    attributes['attributes.RestaurantsAttire.formal'] += \
        attributes['attributes.RestaurantsAttire.uformal']
    attributes['attributes.Smoking.outdoor'] += \
        attributes['attributes.Smoking.uoutdoor']
    attributes['attributes.Smoking.no'] += \
        attributes['attributes.Smoking.uno']
    attributes['attributes.Smoking.yes'] += \
        attributes['attributes.Smoking.uyes']
    attributes['attributes.WiFi.free'] += \
        attributes['attributes.WiFi.ufree']
    attributes['attributes.WiFi.no'] += \
        attributes['attributes.WiFi.uno']
    attributes['attributes.WiFi.paid'] += \
        attributes['attributes.WiFi.upaid']

    # Drop the duplicate columns and save to file
    to_drop = [
        'attributes.Alcohol.ubeer_and_wine',
        'attributes.Alcohol.ufull_bar',
        'attributes.BYOBCorkage.uyes_corkage',
        'attributes.NoiseLevel.uquiet',
        'attributes.NoiseLevel.uaverage',
        'attributes.NoiseLevel.uloud',
        'attributes.NoiseLevel.uvery_loud',
        'attributes.RestaurantsAttire.ucasual',
        'attributes.RestaurantsAttire.udressy',
        'attributes.RestaurantsAttire.uformal',
        'attributes.Smoking.uoutdoor',
        'attributes.Smoking.uno',
        'attributes.Smoking.uyes',
        'attributes.WiFi.ufree',
        'attributes.WiFi.uno',
        'attributes.WiFi.upaid'
    ]
    attributes.drop(to_drop, axis=1, inplace=True)

    print('Saving attributes...')
    with open('data/attribute_names.pickle', 'wb') as f:
        pickle.dump(attributes.columns.to_list(), f)
    attributes = sp.sparse.csr_matrix(attributes.values)
    with open('data/attributes.pickle', 'wb') as f:
        pickle.dump(attributes, f)


def perform_nlp():
    """Performs the NLP steps and saves the matrix and feature names to files.

    This function creates two files: 'data/text.pickle' and
    'data/vocabulary.pickle'. They contain the matrix of word-features and the
    list of column names for those word-features, respectively.

    The reason the NLP is done here is so that the rest of the dataset isn't
    loaded in memory at the same time, since this is one of the most
    computationally intensive steps of the project.
    """
    # Read the cleaned data files and concatenate them into a single DataFrame
    train_text = pd.read_csv('data/train.csv', usecols=['text'])
    test_text = pd.read_csv('data/test.csv', usecols=['text'])
    text = pd.concat([train_text, test_text], ignore_index=False)
    text = text.iloc[:, 0].fillna('').values

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
        return [stemmer.stem(w) for w in regex_tokenizer.tokenize(x.lower())]

    # Fit the TF/IDF vectorizer to the review text data
    tfidf = TfidfVectorizer(strip_accents='ascii',
                            analyzer='word',
                            stop_words=stop_words,
                            tokenizer=tokenizer,
                            max_features=10000)

    print('Performing TF/IDF vectorization (could take >1 hour!)...')
    start = time.time()
    text = tfidf.fit_transform(text)
    end = time.time()
    print('Completed TF/IDF vectorization in {} minutes.'.format(
        (end - start) / 60))

    # Save these objects for later use
    with open('data/text.pickle', 'wb') as f:
        pickle.dump(text, f, pickle.HIGHEST_PROTOCOL)
    with open('data/vocabulary.pickle', 'wb') as f:
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
    # Read the files and concatenate them into a single DataFrame
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    df = pd.concat([
        train.drop(['business_id', 'text', 'stars', 'attributes'], axis=1),
        test.drop(['business_id', 'text', 'KaggleID', 'attributes'], axis=1)
    ], ignore_index=True)

    # Fill remaining missing values and convert data types
    df['is_open'] = df['is_open'].fillna(1)
    df['is_open'] = df['is_open'].astype(np.uint8)
    df['longitude'] = df['longitude'].fillna(df['longitude'].mean())
    df['latitude'] = df['latitude'].fillna(df['latitude'].mean())
    df.iloc[:, 4:] = df.iloc[:, 4:].fillna(0).astype(np.uint8)
    df = df.join(pd.get_dummies(df['state'])).drop(['state'], axis=1)

    # Get the feature names and save them before continuing
    with open('data/attribute_names.pickle', 'rb') as f:
        attribute_names = pickle.load(f)
    with open('data/vocabulary.pickle', 'rb') as f:
        vocabulary = pickle.load(f)
    feature_names = df.columns.to_list() + attribute_names + vocabulary
    with open('data/feature_names.pickle', 'wb') as f:
        pickle.dump(feature_names, f)

    # Join the matrices into a single feature set
    with open('data/text.pickle', 'rb') as f:
        text = pickle.load(f)
    with open('data/attributes.pickle', 'rb') as f:
        attributes = pickle.load(f)

    X = sp.sparse.hstack((sp.sparse.csr_matrix(df.values), attributes, text))

    # Save sparse dataset to files
    sp.sparse.save_npz('data/sparse.npz', X)


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
                                           'data/test.csv']):
        print('The cleaned data needs to be created...')
        make_csv_files()
    if not all(os.path.exists(p) for p in ['data/text.pickle',
                                           'data/vocabulary.pickle']):
        print('The NLP data needs to be created...')
        perform_nlp()
    if not all(os.path.exists(p) for p in ['data/attributes.pickle']):
        print('The attributes need to be extracted...')
        extract_attributes()
    if not all(os.path.exists(p) for p in ['data/sparse.npz',
                                           'data/feature_names.pickle']):
        print('The sparse data matrix needs to be created...')
        make_sparse_matrix()
    print('Done! All the files exist.')


if __name__ == '__main__':
    check_files()
