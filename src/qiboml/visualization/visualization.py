import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
from mpl_toolkits.mplot3d import Axes3D
from qibo import hamiltonians
from qibo.symbols import X, Y, Z


class Bloch:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d", elev=30)

        # Data
        self.states = []
        self.points = []
        self.vectors = []

        # Data
        self.color_states = []
        self.color_points = []
        self.color_vectors = []

    def create_sphere(self):
        "Function to create an empty sphere."

        # Empty sphere
        phi, theta = np.mgrid[0.0 : np.pi : 100j, 0.0 : 2.0 * np.pi : 100j]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        self.ax.plot_surface(x, y, z, color="lavenderblush", alpha=0.2)

        # Circular curves over the surface
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.zeros(100)
        x = np.sin(theta)
        y = np.cos(theta)

        self.ax.plot(x, y, z, color="black", alpha=0.25)
        self.ax.plot(z, x, y, color="black", alpha=0.25)
        self.ax.plot(y, z, x, color="black", alpha=0.1)

        # Axis lines
        line = np.linspace(-1, 1, 100)
        zeros = np.zeros_like(line)

        self.ax.plot(line, zeros, zeros, color="black", alpha=0.3)
        self.ax.plot(zeros, line, zeros, color="black", alpha=0.3)
        self.ax.plot(zeros, zeros, line, color="black", alpha=0.3)

        self.ax.text(1.2, 0, 0, "y", color="black", fontsize=15, ha="center")
        self.ax.text(0, -1.2, 0, "x", color="black", fontsize=15, ha="center")
        self.ax.text(0, 0, 1.2, r"$|0\rangle$", color="black", fontsize=15, ha="center")
        self.ax.text(
            0, 0, -1.2, r"$|1\rangle$", color="black", fontsize=15, ha="center"
        )

        self.ax.set_xlim([-0.8, 0.8])
        self.ax.set_ylim([-0.8, 0.8])
        self.ax.set_zlim([-0.8, 0.8])

    def coordinates(self, state):
        "Function to determine the coordinates of a qubit in the sphere."

        x, y, z = 0, 0, 0
        if state[0] == 1 and state[0] == 0:
            z = 1
        elif state[0] == 0 and state[0] == 1:
            z = -1
        else:
            sigma_X = hamiltonians.SymbolicHamiltonian(X(0))
            sigma_Y = hamiltonians.SymbolicHamiltonian(Y(0))
            sigma_Z = hamiltonians.SymbolicHamiltonian(Z(0))

            x = sigma_X.expectation(state)
            y = sigma_Y.expectation(state)
            z = sigma_Z.expectation(state)
        return x, y, z

    def add_vector(self, x, mode="vector", color="black"):
        if mode == "vector":
            self.color_vectors.append(color)
            self.vectors.append(x)
        elif mode == "point":
            self.color_points.append(color)
            self.points.append(x)
        else:
            RaiseError("Unknown mode. Only 'vector' and 'point' are available.")

    def add_state(self, state, mode="vector", color="black"):
        "Function to add a state to the sphere."

        x, y, z = self.coordinates(state)
        self.color_states.append(color)
        self.states.append(np.array([x, y, z]))

    def rendering(self):

        self.create_sphere()

        for color, vector in zip(self.color_vectors, self.vectors):
            self.ax.quiver(0, 0, 0, *vector, color=color, arrow_length_ratio=0.1)

        for color, state in zip(self.color_vectors, self.states):
            self.ax.quiver(0, 0, 0, *state, color=color, arrow_length_ratio=0.1)

        for color, point in zip(self.color_points, self.points):
            self.ax.scatter(x, y, z, color=color, s=10)

    def plot(self):
        self.ax.set_aspect("equal")
        plt.axis("off")
        plt.tight_layout()
        plt.ion()
        self.fig.show()
