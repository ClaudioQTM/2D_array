import matplotlib.pyplot as plt
from smatrix import t, square_lattice

def plot_dispersion():

    kx_grid = np.linspace(-np.pi/square_lattice.a, np.pi/square_lattice.a, 100)