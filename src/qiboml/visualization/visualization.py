from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from qibo import hamiltonians
from qibo.symbols import X, Y, Z


class Bloch():
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d', elev=30)

    def save_plot(self, name):
        "Function to save the sphere."
        self.ax.set_aspect("equal")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(name)

    def plot(self):
        "Function to plot the sphere."
        self.ax.set_aspect("equal")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


    def coordinates(self, state, mode):    
        "Function to determine the coordinates of a qubit in the sphere." 

        sigma_X = hamiltonians.SymbolicHamiltonian(X(0))
        sigma_Y = hamiltonians.SymbolicHamiltonian(Y(0))
        sigma_Z = hamiltonians.SymbolicHamiltonian(Z(0))
        
        x = sigma_X.expectation(state)
        y = sigma_Y.expectation(state)
        z = sigma_Z.expectation(state)
        return x, y, z
        
        
    def add_vector(self, state, mode, color):
        "Function to add a state to the sphere." 

        x, y, z = 0, 0, 0
        if state[0] == 1 and state[0] == 0:
            z = 1
        if state[0] == 0 and state[0] == 1:
            z = -1
        else:
            x, y, z = self.coordinates(state)

        if mode == "vector":
            vector = np.array([x,y,z])
            self.ax.quiver(0, 0, 0, *vector, color=color, arrow_length_ratio=0.1)
        
        if mode == "point":
            self.ax.scatter(x, y, z, color=color, s=10)
            

    def bloch_sphere(self):
        "Function to create an empty sphere." 

        phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
        x = np.sin(phi)*np.cos(theta)
        y = np.sin(phi)*np.sin(theta)
        z = np.cos(phi)

        self.ax.plot_surface(x, y, z, color='lavenderblush', alpha=0.2)

        # plot circular curves over the surface
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.zeros(100)
        x = np.sin(theta)
        y = np.cos(theta)

        self.ax.plot(x, y, z, color='black', alpha=0.25)
        self.ax.plot(z, x, y, color='black', alpha=0.25)
        self.ax.plot(y, z, x, color='black', alpha=0.1)

        # add axis lines
        line = np.linspace(-1, 1, 100)
        zeros = np.zeros_like(line)

        self.ax.plot(line, zeros, zeros, color='black', alpha=0.3)
        self.ax.plot(zeros, line, zeros, color='black', alpha=0.3)
        self.ax.plot(zeros, zeros, line, color='black', alpha=0.3)

        self.ax.text(1.2, 0, 0, "y", color='black', fontsize=15, ha='center')
        self.ax.text(0, -1.2, 0, "x", color='black', fontsize=15, ha='center')
        self.ax.text(0, 0, 1.2, r"$|0\rangle$", color='black', fontsize=15, ha='center')
        self.ax.text(0, 0, -1.2, r"$|1\rangle$", color='black', fontsize=15, ha='center')
        

        self.ax.set_xlim([-0.8, 0.8])
        self.ax.set_ylim([-0.8, 0.8])
        self.ax.set_zlim([-0.8, 0.8])

        return self



