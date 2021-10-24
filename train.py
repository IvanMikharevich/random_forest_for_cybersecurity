import pandas as pd
from pathlib import Path
import os
import re
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def get_df(dir):
    df = pd.read_csv(dir, sep='\t', header=0, skipinitialspace=True)
    return df

# The find_features function finds unique statically imported exe file
# libraries in the list and saves them to features.txt


def find_features(dir, data_dir):
    df = get_df(dir)
    df = df.copy(deep=True)
    pattern = re.compile("^\s+|\s*,\s*|\s+$")
    for i in range(len(df)):
        string = df['libs'][i]
        df['libs'][i] = [x for x in pattern.split(string) if x]
        unique_numbers = list(set(df['libs'][i]))
        df['libs'][i] = unique_numbers
        if i == 0:
            features = unique_numbers
        else:
            features = list(set(features + unique_numbers))

    with open(data_dir / 'features.txt', 'w') as filehandle:
        for listitem in features:
            filehandle.write('%s\n' % listitem)

# The preprocesing function partially duplicates the find_features function and is needed
# to convert the bliss column into a column with a list of libraries
# unique to this exe file, returns a dataframe


def preprocesing(df):
    _df = df.copy(deep=True)
    pattern = re.compile("^\s+|\s*,\s*|\s+$")

    for i in range(len(_df)):
        string = _df['libs'][i]
        _df['libs'][i] = [x for x in pattern.split(string) if x]
        unique_numbers = list(set(_df['libs'][i]))
        _df['libs'][i] = unique_numbers
    return _df

# The x function reads feature.txt and returns a
# list of unique libraries


def get_list_features():
    features = []
    with open(data_dir / 'features.txt', 'r') as filehandle:
        for line in filehandle:
            currentFeature = line[:-1]
            features.append(currentFeature)
    return features

# In function encoder, the one hot encoding method
# is actually implemented


def encoder(df, y=True):
    features = get_list_features()
    X_array = np.zeros((len(df), len(features)))

    for i in range(len(df)):
        lst1 = df['libs'][i]

        for f in lst1:
            try:
                X_array[i][features.index(f)] = 1
            except ValueError:
                pass

    if y == True:
        Y = df['is_virus']
        Y_array = Y.to_numpy()
        return X_array, Y_array
    else:
        return X_array

# The x function collects all the functions necessary to convert
# a tsv file into vectors or a vector needed for training,
# validation and prediction


def pipline(dir, y=True):
    df = get_df(dir)
    df = preprocesing(df)
    if y==True:
        x_array, y_array = encoder(df)
        return x_array, y_array
    else:
        x_array = encoder(df, y=False)
        return x_array

if __name__ == '__main__':
    workdir = Path(os.getcwd())
    data_dir = workdir / 'data'

    find_features(data_dir / 'train.tsv', data_dir)
    X_train, Y_train = pipline(data_dir / 'train.tsv')


    # In the tasks of binary classification, the Random forest model has proven itself well
    clf = RandomForestClassifier(criterion='entropy', n_estimators=700,
                                 min_samples_split=10, min_samples_leaf=1,
                                 max_features='auto', oob_score=True,
                                 random_state=1,n_jobs=-1)

    clf.fit(X_train, np.ravel(Y_train))
    joblib.dump(clf, "my_random_forest.joblib")


