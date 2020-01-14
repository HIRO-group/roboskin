import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    output = np.linspace(1, 4096, 2)
    plt.plot(output, (output*(6/4096))-3, label="A/C to datasheet")
    plt.plot(output, (output * (10.23 / 4096)) - 5.2, label="Calculated according to accel")
    plt.legend(loc="best")
    plt.show()
