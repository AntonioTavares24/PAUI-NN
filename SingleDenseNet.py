import os
import numpy as np
import pandas as pd
import dltools as dlt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

# Load Jose's data
jose_rest_1 = pd.read_csv('Dataset/Jose-Resting1.txt', sep=';', index_col=False)
jose_rest_2 = pd.read_csv('Dataset/Jose-Resting2.txt', sep=';', index_col=False)
jose_stress_1 = pd.read_csv('Dataset/Jose-Stress1.txt', sep=';', index_col=False)
jose_stress_2 = pd.read_csv('Dataset/Jose-Stress2.txt', sep=';', index_col=False)
jose_wl_1 = pd.read_csv('Dataset/Jose-WL1.txt', sep=';', index_col=False)
jose_wl_2 = pd.read_csv('Dataset/Jose-WL2.txt', sep=';', index_col=False)
jose_rest_1['Class'] = 0
jose_rest_2['Class'] = 0
jose_stress_1['Class'] = 1
jose_stress_2['Class'] = 1
jose_wl_1['Class'] = 2
jose_wl_2['Class'] = 2

# Assemble dataset
jose_data = np.concatenate([jose_rest_1, jose_rest_2, jose_stress_1, jose_stress_2, jose_wl_1, jose_wl_2], axis=0)

# Shuffle dataset
indices = np.arange(jose_data.shape[0])
np.random.shuffle(indices)
jose_data = jose_data[indices]

# Extract labels
jose_labels = jose_data[:, -1]
jose_data = jose_data[:, :-1]

# Split datasets
data_split = int(0.7 * len(jose_data))
jose_train_data = jose_data[:data_split, :]
jose_test_data = jose_data[data_split:, :]
jose_train_labels = jose_labels[:data_split]
jose_test_labels = jose_labels[data_split:]

# Normalize data
mean = jose_train_data.mean(axis=0)
jose_train_data -= mean
std = jose_train_data.std(axis=0)
jose_train_data /= std

train_data = jose_train_data
train_labels = jose_train_labels
test_data = jose_test_data
test_labels = jose_test_labels

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(train_data.shape[1],)))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=Adam(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data,
                    train_labels,
                    epochs=350,
                    batch_size=16,
                    validation_split=0.2)

history_dir = 'History'
models_dir = 'Models'
file_name = 'densenet_jose_v1'
dlt.pickle_dump(history.history, os.path.join(history_dir, file_name))
model.save(models_dir + '/' + file_name + '.h5')

'''
results = model.evaluate(test_data, test_labels)
print(results)
'''
