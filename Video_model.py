import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
import pickle

data = pd.read_csv('Dataset_final.csv')

data = shuffle(data, random_state=0)

x = np.array(data[[str(i) for i in range(1,122)]])
y = np.array(data['122'])

scaler = StandardScaler()

scaler.fit(x)

x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

mlp = MLPClassifier(hidden_layer_sizes=(82,54,36,24,16),activation='relu',solver='sgd',learning_rate_init=0.001,max_iter=2000)

mlp.fit(x_train, y_train)

print(mlp.score(x_test, y_test))

print(mlp.score(x_train, y_train))

file = "Video_model.sav"

pickle.dump(mlp, open(file, 'wb'))