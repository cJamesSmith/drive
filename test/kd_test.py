import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

datas = np.loadtxt("my_waypoint")
xy_data = datas[:, :2]
print(xy_data.shape)

tree = spatial.KDTree(xy_data)
print(tree.data.shape)

# query_p = [-459.89, 338.7]
query_p = xy_data[0]

nearest = tree.query(query_p, 50, p=2)

print(nearest)

plt.plot(xy_data[:, 0], xy_data[:, 1], "o")  # visualize
for near in nearest[1]:
    plt.plot(*xy_data[near], "ro")
# plt.plot([xy_data[near] for near in nearest[1]], "ro")
plt.plot(*query_p, "go")
plt.show()
