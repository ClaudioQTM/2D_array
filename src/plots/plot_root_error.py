import matplotlib.pyplot as plt
import numpy as np

def plot_root_error(results:np.ndarray[complex]):
    error_vec = np.absolute(results-1)
    plt.plot(np.linspace(0, 2*np.pi, 6), error_vec)
    plt.show()

if __name__ == "__main__":
    results = [-0.20185847-1.54623736j, -0.81866531-1.06147992j, -1.12572352-0.80341807j,
    0.50460036-4.16032959j, 1.83696796-1.55285819j, -0.20177247-1.54621384j]

    results = np.asarray(results, dtype=np.complex128)
    plot_root_error(results)