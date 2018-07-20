import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
import pickle

data = pd.read_csv('Image_dataset.csv')

data = shuffle(data, random_state=0)

x = data.loc[:,'0':'399']
y = data['400']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

mlp = MLPClassifier(hidden_layer_sizes=(267, 178, 119, 80, 54, 36),activation='relu',solver='sgd',learning_rate_init=0.001,max_iter=2000)

#training data
mlp.fit(x_train, y_train)

print(mlp.score(x_test, y_test))

print(mlp.score(x_train, y_train))

#dumping model in a file
file = "Gender_model.sav"

pickle.dump(mlp, open(file, 'wb'))
