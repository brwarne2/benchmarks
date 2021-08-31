import numpy as np
# load
data = np.loadtxt("cnae9_train_0.csv", delimiter=',', encoding='utf-8-sig')
print(data.shape)
# don't add noise to classes
labels = data[:, -1]
data = data[:, :-1]
print(labels[:10])
print(data[:10])
# generate noise
noise = np.random.normal(0, .0001, data.shape)
# add noise
new_signal = data + noise
labels = labels.reshape(labels.shape[0], -1)
newData = np.concatenate((new_signal, labels), axis=1)
print(newData[:10])
np.savetxt('n_cnae9_train_0.csv', newData, delimiter=',')