import os
import pandas as pd
import dltools as dlt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

# Load whole dataset
dataset_dir = 'Dataset'

# Load Jorge's test data
jorge_rest = pd.read_csv(os.path.join(dataset_dir, "Jorge-Rest.txt"), sep=';', index_col=False)
jorge_stress = pd.read_csv(os.path.join(dataset_dir, "Jorge-Stress.txt"), sep=';', index_col=False)
jorge_workload = pd.read_csv(os.path.join(dataset_dir, "Jorge-Workload.txt"), sep=';', index_col=False)

# jorge_rest = jorge_rest.loc[:, "Browfurrow":"Upperlipraise"]
# jorge_stress = jorge_stress.loc[:, "Browfurrow":"Upperlipraise"]
# jorge_workload = jorge_workload.loc[:, "Browfurrow":"Upperlipraise"]

# jorge_rest = jorge_rest.loc[:, ["Browfurrow", "Smile"]]
# jorge_stress = jorge_stress.loc[:, ["Browfurrow", "Smile"]]
# jorge_workload = jorge_workload.loc[:, ["Browfurrow", "Smile"]]

jorge_rest['Class'] = 0
jorge_stress['Class'] = 1
jorge_workload['Class'] = 2

# Assemble dataset
jorge_data = np.concatenate([jorge_rest, jorge_stress, jorge_workload], axis=0)

# Shuffle data
indices = np.arange(jorge_data.shape[0])
np.random.shuffle(indices)
jorge_data = jorge_data[indices]

# Extract labels
features = jorge_data[:, :-1]
labels = jorge_data[:, -1]

# Split dataset
data_split = int(0.7 * len(features))
jorge_train_data = features[:data_split, :]
jorge_test_data = features[data_split:, :]
jorge_train_labels = labels[:data_split]
jorge_test_labels = labels[data_split:]

# Normalize data
mean = jorge_train_data.mean(axis=0)
std = jorge_train_data.std(axis=0)
jorge_train_data -= mean
jorge_train_data /= std

jorge_test_data -= mean
jorge_test_data /= std

# Model definition
train_data = jorge_train_data
train_labels = jorge_train_labels
test_data = jorge_test_data
test_labels = jorge_test_labels

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(train_data.shape[1],)))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=Adam(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data,
                    train_labels,
                    epochs=80,
                    batch_size=32,
                    validation_split=0.2)

history_dir = 'History'
models_dir = 'Models'
file_name = 'test_net_paui_v2'
dlt.pickle_dump(history.history, os.path.join(history_dir, file_name))
model.save(models_dir + '/' + file_name + '.h5')

results = model.evaluate(test_data, test_labels)
print(results)