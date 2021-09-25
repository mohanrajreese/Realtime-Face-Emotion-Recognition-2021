import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
print("Ignored the warnings and read the dataset")
data = pd.read_csv('./fer2013.csv')
width, height = 48, 48


print("converting a column in a dataframe into a list")
datapoints = data['pixels'].tolist()

print("getting features for training")
X = []
for i in datapoints:
    xx = [int(xp) for xp in i.split(' ')]
    xx = np.asarray(xx).reshape(width, height)
    X.append(xx.astype('float32'))

print("converting list of data's to array")
X = np.asarray(X)
X = np.expand_dims(X, -1)

print("getting labels for training")
y = pd.get_dummies(data['emotion']).to_numpy()

print(" storing them using numpy")
np.save('data', X)
np.save('labels', y)

print("Preprocessing Done")
print("Number of Features: " + str(len(X[0])))
print("Number of Labels: " + str(len(y[0])))
print("Number of examples in dataset:" + str(len(X)))
print("X,y stored in data.npy and labels.npy respectively")
