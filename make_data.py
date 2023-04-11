import numpy as np

data = np.random.rand(100, 3)
np.savetxt("data.csv", data, delimiter=",")