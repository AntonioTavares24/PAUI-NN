import numpy as np
from keras.models import load_model

model = load_model('Models/test_net_paui.h5')
layer_weights = []

for layer in model.layers:
    layer_weights.append(layer.get_weights())

layer_weights = np.asarray(layer_weights)

relu1_weights = layer_weights[0, 0]
relu1_bias = layer_weights[0, 1]
softmax_weights = layer_weights[1, 0]
softmax_bias = layer_weights[1, 1]

np.savetxt("D:/PAUI-(M-ITI)/Network Weights/relu1_weights.csv", relu1_weights, delimiter=";")
np.savetxt("D:/PAUI-(M-ITI)/Network Weights/relu1_bias.csv", relu1_bias, delimiter=";")
np.savetxt("D:/PAUI-(M-ITI)/Network Weights/softmax_weights.csv", softmax_weights, delimiter=";")
np.savetxt("D:/PAUI-(M-ITI)/Network Weights/softmax_bias.csv", softmax_bias, delimiter=";")
