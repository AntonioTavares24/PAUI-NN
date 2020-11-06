import os
import numpy as np
import pandas as pd
import dltools as dlt

user_name = 'User1'
save_dir = 'Objects'

# A pasta signals_dir contém os ficheiros com os dados recolhidos
# Nesta pasta, os nomes dos ficheiros das três tarefas (rest, stress e workload)
# devem estar organizados alfabeticamente por esta ordem. Por exemplo,
# se os nomes forem User5_Rest, User5_Stress e User5_Workload está tudo bem

signals_dir = 'C:/Users/User/Desktop/PAUI-(M-ITI)/Sample-Signals'

signals_list = os.listdir(os.path.join(signals_dir, user_name))

rest_path = os.path.join(signals_dir, user_name, signals_list[0])
stress_path = os.path.join(signals_dir, user_name, signals_list[1])
workload_path = os.path.join(signals_dir, user_name, signals_list[2])

rest_df = pd.read_csv(rest_path, sep=';', index_col=False)
stress_df = pd.read_csv(stress_path, sep=';', index_col=False)
workload_df = pd.read_csv(workload_path, sep=';', index_col=False)

rest_df = rest_df.reset_index(drop=True)
stress_df = stress_df.reset_index(drop=True)
workload_df = workload_df.reset_index(drop=True)

del rest_df['HourMinSec']
del stress_df['HourMinSec']
del workload_df['HourMinSec']

del rest_df['PEAKS']
del stress_df['PEAKS']
del workload_df['PEAKS']

rest_df['Class'] = 0
stress_df['Class'] = 1
workload_df['Class'] = 2

data_split = int(0.75 * len(rest_df))
rest_train_df = rest_df.loc[:data_split, :].reset_index(drop=True)
rest_test_df = rest_df.loc[data_split:, :].reset_index(drop=True)

data_split = int(0.75 * len(stress_df))
stress_train_df = stress_df.loc[:data_split, :].reset_index(drop=True)
stress_test_df = stress_df.loc[data_split:, :].reset_index(drop=True)

data_split = int(0.75 * len(workload_df))
workload_train_df = workload_df.loc[:data_split, :].reset_index(drop=True)
workload_test_df = workload_df.loc[data_split:, :].reset_index(drop=True)

train_data = np.concatenate([rest_train_df, stress_train_df, workload_train_df], axis=0).astype('float64')
test_data = np.concatenate([rest_test_df, stress_test_df, workload_test_df], axis=0).astype('float64')

train_labels = train_data[:, -1]
train_data = train_data[:, :-1]
test_labels = test_data[:, -1]
test_data = test_data[:, :-1]

dlt.pickle_dump(train_data, os.path.join(save_dir, user_name + '_train_data.obj'))
dlt.pickle_dump(train_labels, os.path.join(save_dir, user_name + '_train_labels.obj'))
dlt.pickle_dump(test_data, os.path.join(save_dir, user_name + '_test_data.obj'))
dlt.pickle_dump(test_labels, os.path.join(save_dir, user_name + '_test_labels.obj'))
