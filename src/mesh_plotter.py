from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class MeshPlotter:
    def __init__(self, laplace_mesh):
        self.__laplace_mesh = laplace_mesh
        npx, npy = laplace_mesh.get_n_points()
        self.__npt_x = npx
        self.__npt_y = npy

    def plot_potential(self, potential_grid):
        plt.matshow(potential_grid)
        plt.show()

    def plot_electric_field(self, field_x_grid, field_y_grid, field_abs_grid):
        x, y = range(self.__npt_x), range(self.__npt_y)
        # Use arrows to show directions of the field.
        # Swapped because of matrix row/colum geometry
        plt.quiver(x, y, field_y_grid, field_x_grid)
        plt.title("Electric Field Vector")
        plt.matshow(field_abs_grid)
        plt.title("Electric Field Magnitude")
        plt.show()

    def plot_potential_profile(self, x_axis, potential_profile, scaled=False):
        """
        Displays the potential profile of the coaxial cable along the
        horizontal axis of symmetry.
        @param scaled: Scale values by the cable width.
        @return:
        """
        plt.plot(x_axis, potential_profile)
        if scaled:
            plt.xlabel("Relative position along cable")
        else:
            plt.xlabel("X position along cable cross-section (cm)")
        plt.ylabel("Potential (Volts)")
        plt.grid(b=True, which='both', axis='both', color='k')
        plt.show()

    def visualise_3d(self, point_dict, display='V'):
        """
        3-dimensional visualisation tool for electrostatic quantities:
        @param display: if this is set to,'E' then shows the variation of
                        electric field magnitude, else if 'V', displays
                        the potential
        @return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(range(self.__npt_x), range(self.__npt_y))
        potential_matrix = self.__laplace_mesh.get_potential_grid(plot=False)
        if display == 'V':
            ax.plot_surface(y, x, potential_matrix, rstride=1, cstride=1,
                            cmap=cm.Spectral, linewidth=0, antialiased=False)
        elif display == 'E':
            e_field_matrix = self.__laplace_mesh.self.get_electric_field_grid()
            ax.plot_surface(y, x, e_field_matrix, rstride=1, cstride=1,
                            cmap=cm.Spectral, linewidth=0, antialiased=False)
        else:
            raise Exception("'E' or 'V' must be inputted for display.")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("V (V)" if display == 'V' else "E (V/cm)")

