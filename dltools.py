import os
import pickle
import pandas as pd


def load_csv_data(file_name, directory):
    file_path = os.path.join(directory, file_name)
    dataframe = pd.read_csv(file_path)
    data = dataframe.values
    columns = dataframe.columns.values
    return data, columns


def pickle_dump(file, path):
    with open(path, 'wb') as out_file:
        pickle.dump(file, out_file)


def pickle_load(path):
    with open(path, 'rb') as in_file:
        loaded_object = pickle.load(in_file)
    return loaded_object
