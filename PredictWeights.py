import numpy as np
from keras.models import load_model

user_name = 'User1'
version = '1.0'

model = load_model('Models/m_dense_kfold_' + user_name + '_v' + version + '.h5')
layer_weights = []

for layer in model.layers:
    layer_weights.append(layer.get_weights())

layer_weights = np.asarray(layer_weights)

# Deve existir um _weights e um _bias por cada layer que a network tem (neste caso terá três)
dense1_weights = layer_weights[0, 0]
dense1_bias = layer_weights[0, 1]
dense2_weights = layer_weights[1, 0]
dense2_bias = layer_weights[1, 1]
dense3_weights = layer_weights[2, 0]
dense3_bias = layer_weights[2, 1]
softmax_weights = layer_weights[3, 0]
softmax_bias = layer_weights[3, 1]

# Igualmente, aqui tem que haver um par de saves por layer
np.savetxt("D:/PAUI-(M-ITI)/Network Weights/" + user_name + "_dense1_weights.csv", dense1_weights, delimiter=";")
np.savetxt("D:/PAUI-(M-ITI)/Network Weights/" + user_name + "_dense1_bias.csv", dense1_bias, delimiter=";")
np.savetxt("D:/PAUI-(M-ITI)/Network Weights/" + user_name + "_dense2_weights.csv", dense2_weights, delimiter=";")
np.savetxt("D:/PAUI-(M-ITI)/Network Weights/" + user_name + "_dense2_bias.csv", dense2_bias, delimiter=";")
np.savetxt("D:/PAUI-(M-ITI)/Network Weights/" + user_name + "_dense3_weights.csv", dense3_weights, delimiter=";")
np.savetxt("D:/PAUI-(M-ITI)/Network Weights/" + user_name + "_dense3_bias.csv", dense3_bias, delimiter=";")
np.savetxt("D:/PAUI-(M-ITI)/Network Weights/" + user_name + "_softmax_weights.csv", softmax_weights, delimiter=";")
np.savetxt("D:/PAUI-(M-ITI)/Network Weights/" + user_name + "_softmax_bias.csv", softmax_bias, delimiter=";")
