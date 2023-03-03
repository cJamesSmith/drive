import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('my_waypoint')

plt.plot(data[:, 0], data[:, 1], 'o')
plt.show()