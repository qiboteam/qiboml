import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch

from qibo import hamiltonians
from qibo.symbols import X, Y, Z


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


class Bloch:
    def __init__(self, state=None, vector=None, point=None):
        # Sizes
        self.figsize = [5, 5]
        self.fontsize = 18
        self.arrow_style = "-|>"
        self.arrow_width = 2.0
        self.mutation_scale = 20

        # Figure and axis
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection="3d", elev=30, azim=30)

        # Bool variable
        self._shown = False

        # Data
        self.states = []
        self.points = []
        self.vectors = []

        # Color data
        self.color_states = []
        self.color_points = []
        self.color_vectors = []

    def clear(self):
        # Data
        self.states = []
        self.points = []
        self.vectors = []

        # Color data
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

        # ----Circular curves over the surface----
        # Axis
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.zeros(100)
        x = np.sin(theta)
        y = np.cos(theta)
        self.ax.plot(x, y, z, color="black", alpha=0.25)
        self.ax.plot(z, x, y, color="black", alpha=0.25)
        self.ax.plot(y, z, x, color="black", alpha=0.1)

        # Latitude
        z1 = np.full(100, 0.4)
        r1 = np.sqrt(1 - z1[0] ** 2)
        x1 = r1 * np.cos(theta)
        y1 = r1 * np.sin(theta)
        self.ax.plot(x1, y1, z1, color="black", alpha=0.1)

        z1 = np.full(100, 0.9)
        r1 = np.sqrt(1 - z1[0] ** 2)
        x1 = r1 * np.cos(theta)
        y1 = r1 * np.sin(theta)
        self.ax.plot(x1, y1, z1, color="black", alpha=0.1)

        z2 = np.full(100, -0.9)
        r2 = np.sqrt(1 - z2[0] ** 2)
        x2 = r2 * np.cos(theta)
        y2 = r2 * np.sin(theta)
        self.ax.plot(x2, y2, z2, color="black", alpha=0.1)

        # Longitude
        phi_list = np.linspace(0, 2 * np.pi, 6)
        theta = np.linspace(0, 2 * np.pi, 100)

        for phi in phi_list:
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            self.ax.plot(x, y, z, color="black", alpha=0.1)

        # ----Axis lines----
        line = np.linspace(-1, 1, 100)
        zeros = np.zeros_like(line)

        self.ax.plot(line, zeros, zeros, color="black", alpha=0.3)
        self.ax.plot(zeros, line, zeros, color="black", alpha=0.3)
        self.ax.plot(zeros, zeros, line, color="black", alpha=0.3)

        self.ax.text(1.2, 0, 0, "x", color="black", fontsize=self.fontsize, ha="center")
        self.ax.text(0, 1.2, 0, "y", color="black", fontsize=self.fontsize, ha="center")
        self.ax.text(
            0,
            0,
            1.2,
            r"$|0\rangle$",
            color="black",
            fontsize=self.fontsize,
            ha="center",
        )
        self.ax.text(
            0,
            0,
            -1.3,
            r"$|1\rangle$",
            color="black",
            fontsize=self.fontsize,
            ha="center",
        )

        self.ax.set_xlim([-0.7, 0.7])
        self.ax.set_ylim([-0.7, 0.7])
        self.ax.set_zlim([-0.7, 0.7])

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

    def add_vector(self, elements, mode="vector", color="black"):

        if isinstance(elements, list) and isinstance(color, list):
            if len(elements) != len(color):
                raise ValueError("Elements and color must be lists of the same length")
        if not isinstance(elements, list):
            elements = [elements]
        if not isinstance(color, list):
            color = [color] * len(elements)

        if mode == "vector":
            for c, element in zip(color, elements):
                self.color_vectors.append(c)
                self.vectors.append(element)
        elif mode == "point":
            for c, element in zip(color, elements):
                self.color_points.append(c)
                self.points.append(element)
        else:
            RaiseError("Unknown mode. Only 'vector' and 'point' are available.")

    def add_state(self, states, color="black"):
        "Function to add a state to the sphere."
        if isinstance(states, list) and isinstance(color, list):
            if len(states) != len(color):
                raise ValueError("States and color must be lists of the same length")
        if not isinstance(states, list):
            states = [states]
        if not isinstance(color, list):
            color = [color] * len(states)

        for c, state in zip(color, states):
            x, y, z = self.coordinates(state)
            self.color_states.append(c)
            self.states.append(np.array([x, y, z]))

    def rendering(self):
        if self._shown == True:
            plt.close(self.fig)
            self.fig = plt.figure(figsize=self.figsize)
            self.ax = self.fig.add_subplot(111, projection="3d", elev=30, azim=30)

        self._shown = True

        self.create_sphere()

        for color, vector in zip(self.color_vectors, self.vectors):
            xs3d = vector[0] * np.array([0, 1])
            ys3d = vector[1] * np.array([0, 1])
            zs3d = vector[2] * np.array([0, 1])
            a = Arrow3D(
                xs3d,
                ys3d,
                zs3d,
                lw=self.arrow_width,
                arrowstyle=self.arrow_style,
                mutation_scale=self.mutation_scale,
                color=color,
            )
            self.ax.add_artist(a)

        for color, state in zip(self.color_states, self.states):
            xs3d = state[0] * np.array([0, 1])
            ys3d = state[1] * np.array([0, 1])
            zs3d = state[2] * np.array([0, 1])
            a = Arrow3D(
                xs3d,
                ys3d,
                zs3d,
                lw=self.arrow_width,
                arrowstyle=self.arrow_style,
                mutation_scale=self.mutation_scale,
                color=color,
            )
            self.ax.add_artist(a)

        for color, point in zip(self.color_points, self.points):
            self.ax.scatter(point[0], point[1], point[2], color=color, s=10)

    def plot(self):
        self.rendering()
        self.ax.set_aspect("equal")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
