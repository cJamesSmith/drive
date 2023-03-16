import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    state_and_control = np.loadtxt(os.path.join(os.path.dirname(__file__), "state_and_control"))
    print(state_and_control.shape)
    time_span = range(len(state_and_control))
    plt.plot(time_span, state_and_control[:, 0], time_span, state_and_control[:, -1])
    plt.legend(["err_d", "steer"])
    plt.show()
