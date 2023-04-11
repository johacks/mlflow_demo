from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

data = np.loadtxt("data.csv", delimiter=",")

X = data[:, :2]
y = data[:, 2]

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
