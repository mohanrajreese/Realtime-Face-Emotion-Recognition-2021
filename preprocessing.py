import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv('./fer2013.csv')

width, height = 48, 48


# convert a column in a dataframe into a list
datapoints = data['pixels'].tolist()

# getting features for training
X = []
for i in datapoints:
    xx = [int(xp) for xp in i.split(' ')]
    xx = np.asarray(xx).reshape(width, height)
    X.append(xx.astype('float32'))

X = np.asarray(X)
X = np.expand_dims(X, -1)

# getting labels for training
y = pd.get_dummies(data['emotion']).to_numpy()

# storing them using numpy
np.save('data', X)
np.save('labels', y)

print("Preprocessing Done")
print("Number of Features: " + str(len(X[0])))
print("Number of Labels: " + str(len(y[0])))
print("Number of examples in dataset:" + str(len(X)))
print("X,y stored in data.npy and labels.npy respectively")
