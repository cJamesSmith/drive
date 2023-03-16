import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    states = np.loadtxt(os.path.join(os.path.dirname(__file__), "states"))
    # states = states[100:1800]
    actions = np.loadtxt(os.path.join(os.path.dirname(__file__), "actions"))
    print(states.shape)
    time_span = range(len(states))
    plt.plot(time_span, states[:, 0])
    plt.legend(["err_d", "steer"])
    plt.show()
