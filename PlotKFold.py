import os
import dltools as dlt
import matplotlib.pyplot as plt

user_name = 'User1'
version = '1.0'
load_dir = 'Objects'

load_dir = 'History'
save_dir = 'Graphs/'
file_name = 'dense_kfold_' + user_name + '_v' + version

average_history = dlt.pickle_load(os.path.join(load_dir, file_name))
acc = average_history[0]
val_acc = average_history[1]
loss = average_history[2]
val_loss = average_history[3]
epochs = range(1, len(acc) + 1)

plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig, ax = plt.subplots(nrows=1, ncols=2)

ax[0].plot(epochs, acc, '-', color='tab:blue', linewidth='2', label='Training accuracy')
ax[0].plot(epochs, val_acc, 'r-', color='tab:red', linewidth='2', label='Validation accuracy')
ax[0].set_xlabel("Epochs", labelpad=10, fontsize=10, color="#333533")
ax[0].set_ylabel("Accuracy", labelpad=10, fontsize=10, color="#333533")
ax[1].plot(epochs, loss, '-', color='tab:blue', linewidth='2', label='Training loss')
ax[1].plot(epochs, val_loss, '-', color='tab:red', linewidth='2', label='Validation loss')
ax[1].set_xlabel("Epochs", labelpad=10, fontsize=10, color="#333533")
ax[1].set_ylabel("Loss", labelpad=10, fontsize=10, color="#333533")

ax[0].grid(True, color="#93a1a1", alpha=0.3)
ax[0].legend(prop={'size':8})
ax[1].grid(True, color="#93a1a1", alpha=0.3)
ax[1].legend(prop={'size':8})

ax[0].set_facecolor('whitesmoke')
ax[1].set_facecolor('whitesmoke')
plt.gcf().subplots_adjust(bottom=0.15, wspace=0.35)
plt.show()

fig.savefig(save_dir + file_name + '.png')

print(max(val_acc))
