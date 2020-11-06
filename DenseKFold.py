import os
import numpy as np
import dltools as dlt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import regularizers
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

user_name = 'User1'
version = '1.0'

enable_pca = False
pca_factor = 0.95
kfold_val = True
lr = 0.0001
epochs = 300
bs = 32

load_dir = 'Objects'

train_data = dlt.pickle_load(os.path.join(load_dir, user_name + '_train_data.obj'))
train_labels = dlt.pickle_load(os.path.join(load_dir, user_name + '_train_labels.obj'))
test_data = dlt.pickle_load(os.path.join(load_dir, user_name + '_test_data.obj'))
test_labels = dlt.pickle_load(os.path.join(load_dir, user_name + '_test_labels.obj'))


def build_model():
    if kfold_val:
        shape = (partial_train_data.shape[1],)
    else:
        shape = (train_data.shape[1],)

    model = Sequential()
    model.add(Dense(40, activation='relu', input_shape=shape))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=Adam(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if kfold_val:
    k = 4
    num_epochs = epochs
    num_val_samples = len(train_data) // k
    all_histories = [[], [], [], []]
    best_accuracy = []

    skf = StratifiedKFold(n_splits=k)

    for index, (train_indices, val_indices) in enumerate(skf.split(train_data, train_labels)):
        print("Training on fold " + str(index+1) + "/4...")

        partial_train_data, val_data = train_data[train_indices], train_data[val_indices]
        partial_train_labels, val_labels = train_labels[train_indices], train_labels[val_indices]

        mean = partial_train_data.mean(axis=0)
        partial_train_data -= mean
        std = partial_train_data.std(axis=0)
        partial_train_data /= std

        val_data -= mean
        val_data /= std

        if enable_pca:
            pca = PCA(pca_factor)
            pca.fit(partial_train_data)

            partial_train_data = pca.transform(partial_train_data)
            val_data = pca.transform(val_data)

        indices = np.arange(partial_train_data.shape[0])
        np.random.shuffle(indices)
        partial_train_data = partial_train_data[indices]
        partial_train_labels = partial_train_labels[indices]

        partial_train_labels = to_categorical(partial_train_labels)
        val_labels = to_categorical(val_labels)

        model = build_model()
        history = model.fit(partial_train_data,
                            partial_train_labels,
                            validation_data=(val_data, val_labels),
                            epochs=num_epochs,
                            batch_size=bs)

        acc_history = history.history['acc']
        val_acc_history = history.history['val_acc']
        loss_history = history.history['loss']
        val_loss_history = history.history['val_loss']
        all_histories[0].append(acc_history)
        all_histories[1].append(val_acc_history)
        all_histories[2].append(loss_history)
        all_histories[3].append(val_loss_history)
        best_accuracy.append(max(val_acc_history))

    average_history = [[np.mean([x[i] for x in all_histories[j]]) for i in range(num_epochs)] for j in range(4)]
    print(best_accuracy)
    print(np.mean(np.array(best_accuracy)))

    history_dir = 'History'

    if enable_pca:
        file_name = 'dense_kfold_' + user_name + '_pca' + '_v' + version
    else:
        file_name = 'dense_kfold_' + user_name + '_v' + version

    dlt.pickle_dump(average_history, os.path.join(history_dir, file_name))

else:
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std

    test_data -= mean
    test_data /= std

    dlt.pickle_dump(mean, os.path.join(load_dir, user_name + '_mean.obj'))
    dlt.pickle_dump(std, os.path.join(load_dir, user_name + '_std.obj'))

    if enable_pca:
        pca = PCA(pca_factor)
        pca.fit(train_data)

        dlt.pickle_dump(pca.components_, os.path.join(load_dir, user_name + '_pca' + '_v' + version + '.obj'))

        train_data = pca.transform(train_data)
        test_data = pca.transform(test_data)

    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data = train_data[indices]
    train_labels = train_labels[indices]

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    model = build_model()

    model.compile(optimizer=Adam(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs, batch_size=bs)

    loss_score, acc_score = model.evaluate(test_data, test_labels)
    print("Accuracy on test set:", acc_score)

    if enable_pca:
        file_name = 'dense_kfold_' + user_name + '_pca' + '_v' + version
    else:
        file_name = 'dense_kfold_' + user_name + '_v' + version
    model.save('Models/' + file_name + '.h5')
