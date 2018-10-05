import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Dataset_1_sec3.csv')

from sklearn.utils import shuffle
data = shuffle(data, random_state=0)

x = np.array(data[[str(i) for i in range(0,120)]])
y = np.array(data['120'])

scaler = StandardScaler()
scaler.fit(x)

x = scaler.transform(x)

from sklearn.decomposition import NMF
model = NMF(n_components=60, init='random', random_state=0)
x[x<0] = 0

model.fit(x)

x = model.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(82,54,36,24,16),activation='relu',solver='sgd',learning_rate_init=0.001,max_iter=4000)

mlp.fit(x_train, y_train)

print(mlp.score(x_test, y_test))

print(mlp.score(x_train, y_train))

file = "Video_model.sav"

pickle.dump(mlp, open(file, 'wb'))

