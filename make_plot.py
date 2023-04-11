import pickle
import matplotlib.pyplot as plt
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

data = np.loadtxt("data.csv", delimiter=",")

y = data[:, 2]
X = data[:, :2]

y_pred = model.predict(X)

plt.scatter(y, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.savefig("plot.png")
