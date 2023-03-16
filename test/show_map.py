import matplotlib.pyplot as plt
import numpy as np
import os

data = np.loadtxt(os.path.join(os.path.dirname(__file__), "my_waypoint"))

plt.plot(data[:, 0], data[:, 1], "o")
plt.show()
