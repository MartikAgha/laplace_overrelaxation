from itertools import product

import numpy as np
from mesh_plotter import MeshPlotter


# Module used to simulate the 2-dimensional setup of a square coaxial cable cross
# section. This is done using finite difference approximation and is done with
# input parameters relating to the potential, cable dimensions in millimetres, and
# the number of mesh points per millimetre.

class LaplaceMesh:

    default_omega_range = np.linspace(1.01, 1.9, 30)

    def __init__(self, Lx, Ly, wbar=2, ppcm=2, V=10):
        """
        Class to implement the solution of the potential profile of a square coaxial
        cable in 2 dimensions:
        @param Lx: Width of outer tube (millimetres)
        @param Ly: Height of outer tube (millimetres)
        @param wbar: Width of square central bar in coaxial cable (millimetres).
        @param ppcm: Number of square mesh points per centimetre.
        @param V: Electrostatic potential of central bar (volts)
        """
        self.__tube_width = Lx
        self.__tube_height = Ly
        self.__ppcm = ppcm
        self.__potential = V
        self.__bar_width = wbar
        # Build grid with distances and density, adding 1 for the far edge
        self.__npt_x = Lx * ppcm + 1
        self.__npt_y = Ly * ppcm + 1
        # Dictionary to store points and retrieve info quickly
        self.__point_dict = {}
        # Bool matrix for points in the grid that are fixed.
        self.__fixed_points = np.zeros((self.__npt_x, self.__npt_y))
        # Initialises zeros as guesses.
        for i, j in product(range(self.__npt_x), range(self.__npt_y)):
            self.__point_dict[(i, j)] = 0
            self.__fixed_points[i, j] = False
        # Employ fixed boundary conditions
        for i in range(self.__npt_x):
            self.__fixed_points[i, 0] = True
            self.__fixed_points[i, self.__npt_y - 1] = True
        for j in range(self.__npt_y):
            self.__fixed_points[0, j] = True
            self.__fixed_points[self.__npt_x - 1, j] = True
        # Inner bar is a fixed equipotential at given potential
        range_x = range(int(self.__npt_x / 2 - ppcm * wbar / 2),
                        int(self.__npt_x / 2 + ppcm * wbar / 2) + 1)
        range_y = range(int(self.__npt_y / 2 - ppcm * wbar / 2),
                        int(self.__npt_y / 2 + ppcm * wbar / 2) + 1)
        for i, j in product(range_x, range_y):
            self.__fixed_points[i, j] = True
            self.__point_dict[(i, j)] = V

        self.__plotter = MeshPlotter(self)

    def get_n_points(self):
        return self.__npt_x, self.__npt_y

    def get_laplace_stencil(self, i, j, point_dict):
        """
        Obtain the laplacian stencil in this model
        @param i: row index of grid
        @param j: column index of grid
        @param point_dict: copy of current dictionary of points to calculate
                           the new values of these points
        @return: stencil: Value of the new (i, j)
        """
        term1 = point_dict[(i + 1, j)] + point_dict[(i - 1, j)]
        term2 = point_dict[(i, j + 1)] + point_dict[(i, j - 1)]
        stencil = (term1 + term2)/4
        return stencil

    def get_laplace_stencil_or(self, i, j, omega):
        """
        Obtain the laplacian stencil for successive over-relaxation
        @param i: row index of grid
        @param j: column index of grid
        @param omega: value of the relaxation parameter
        @return: stencil: Value of the new (i, j)
        """
        term1 = self.__point_dict[(i + 1, j)] + self.__point_dict[(i - 1, j)]
        term2 = self.__point_dict[(i, j + 1)] + self.__point_dict[(i, j - 1)]
        term3 = self.__point_dict[(i, j)]
        stencil = (0.25*omega)*(term1 + term2) - (omega - 1)*term3
        return stencil

    def solve_with_jacobi(self, tolerance=1e-8, max_iter=10e8):
        """
        Implements Jacobi relaxation method to iteratively obtain solution:
        @param tolerance: Fractional convergence requirement of the
                        2-norm of the solution vector.
        @param max_iter: In the instance of slow convergence, ceases
                         iterative procedure. Usually higher than any
                         reasonable number of iterations.
        @return: iter_count: Total number of iterations performed.
        """
        # The l=2 norm of the solution vector is accumulated throughout
        iter_count = 0
        current_norm = 0.
        while iter_count < max_iter:
            iter_count += 1
            solution_mag = 0
            point_dict_copy = self.__point_dict.copy()
            for i, j in product(range(1, self.__npt_x), range(1, self.__npt_y)):
                if not self.__fixed_points[i, j]:
                    # Laplacian stencil operates on non-fixed points.
                    stencil = self.get_laplace_stencil(i, j, point_dict_copy)
                    self.__point_dict[(i, j)] = stencil
                    solution_mag += stencil** 2
            previous_norm, current_norm = current_norm, np.sqrt(solution_mag)
            # Convergence is declared upon the fractional difference between
            # the 2-norm of the solution vector of the previous and current
            # iteration being less than the given epsilon.
            if np.abs(previous_norm - current_norm) < previous_norm*tolerance:
                return iter_count
        return max_iter

    def solve_with_gauss_seidel(self, tolerance=1e-8, max_iter=10e8):
        """
        Implements Gauss-Seidel relaxation method to iteratively obtain
        solution. Instead of using an entire copy of the previous point dict
        to construct the new iteration, the updated points so far are used in
        the construction of the next iteration's point dict.
        @param tolerance: Fractional convergence requirement of the
                        2-norm of the solution vector.
        @param max_iter: In the instance of slow convergence, ceases
                         iterative procedure. Usually higher than any
                         reasonable number of iterations.
        @return: iter_count: Total number of iterations performed.
        """
        # The l=2 norm of the solution vector is accumulated throughout
        iter_count = 0
        current_norm = 0.
        while iter_count < max_iter:
            iter_count += 1
            solution_mag = 0
            for i, j in product(range(1, self.__npt_x), range(1, self.__npt_y)):
                if not self.__fixed_points[i, j]:
                    # Laplacian stencil operates on non-fixed points.
                    stencil = self.get_laplace_stencil(i, j, self.__point_dict)
                    self.__point_dict[(i, j)] = stencil
                    solution_mag += stencil**2
            previous_norm, current_norm = current_norm, np.sqrt(solution_mag)
            # Convergence is declared upon the fractional difference between
            # the 2-norm of the solution vector of the previous and current
            # iteration being less than the given epsilon.
            if np.abs(previous_norm - current_norm) < previous_norm*tolerance:
                return iter_count
        return max_iter

    def solve_with_successive_over_relaxation(self,
                                              omega=None,
                                              tolerance=1e-8,
                                              max_iter=10e8):
        """
        Implements Successive Over-Relation method to obtain solution.
        @param omega: Relaxation factor to optimise the
                      convergence speed. If None then assumes the
                      theoretical value.
        @param tolerance: Fractional convergence requirement of the
                          2-norm of the solution vector.
        @param max_iter: In the instance of slow convergence, ceases
                         iterative procedure. Usually higher than any
                         reasonable number of iterations.
        @return: iter_count: Total number of iterations performed.
        """
        if omega is None:
            omega = self.get_theoretical_relaxation_factor()
        # The l=2 norm of the solution vector is accumulated throughout
        iter_count = 0
        current_norm = 0
        while iter_count < max_iter:
            iter_count += 1
            solution_mag = 0
            for i, j in product(range(1, self.__npt_x), range(1, self.__npt_y)):
                if not self.__fixed_points[i, j]:
                    stencil_or = self.get_laplace_stencil_or(i, j, omega)
                    self.__point_dict[(i, j)] = stencil_or
                    solution_mag += stencil_or**2
            previous_norm, current_norm = current_norm, np.sqrt(solution_mag)
            """
            Convergence is declared upon the fractional difference between
            the 2-norm of the solution vector of the previous and current
            iteration being less than the given epsilon.
            """
            if np.abs(previous_norm - current_norm) < previous_norm*tolerance:
                return iter_count
        return max_iter
    
    def get_potential_grid(self, plot=False):
        """
        Creates a matrix to either visually plot the potential or use
        values along a given axis.
        @param plot: Plot the grid using matshow
        @return: potential_grid
        """
        potential_grid = np.zeros((self.__npt_x, self.__npt_y))
        for i, j in product(range(self.__npt_x), range(self.__npt_y)):
            potential_grid[i, j] = self.__point_dict[(i, j)]
        if plot:
            self.__plotter.plot_potential(potential_grid)
            # plt.matshow(potential_grid)
            # plt.show()
        return potential_grid

    def get_electric_field(self, i, j, h, direction='x'):
        """
        Get the first order finite difference of the electrostatic potential
        @param i: row index of point grid
        @param j: column index of point grid
        @param h: physical distance between neighbouring grid points
        @param direction: direction 'x' or 'y' to calculate gradient along
        @return: negative_gradient: -gradient of potential
        """
        if direction == 'x':
            fwd_value = self.__point_dict[(i + 1, j)]
            bwd_value = self.__point_dict[(i - 1, j)]
            negative_gradient = -(fwd_value - bwd_value)/(2*h)
            return negative_gradient
        elif direction == 'y':
            fwd_value = self.__point_dict[(i, j + 1)]
            bwd_value = self.__point_dict[(i, j - 1)]
            negative_gradient = -(fwd_value - bwd_value)/(2*h)
            return negative_gradient
        else:
            raise ValueError("Supply either 'x' or 'y' for the direction.")

    def get_electric_field_grid(self, plot=False):
        """
        Creates a matrix to either visually plot the electric_field or use
        values along a given axis.
        @param plot: Plot the grid using matshow
        @return: potential_grid
        """
        field_x_grid = np.zeros((self.__npt_x, self.__npt_y))
        field_y_grid = np.zeros((self.__npt_x, self.__npt_y))
        # At the fixed boundaries the field is of zero magnitude since the
        # surface is a conducting equipotential.
        field_abs_grid = np.zeros((self.__npt_x, self.__npt_y))
        # This scaling ensures the value of the E field is in units of V/cm
        h = self.__ppcm ** (-1)
        for i, j in product(range(self.__npt_x), range(self.__npt_y)):
            if not self.__fixed_points[i, j]:
                field_x_grid[i, j] = self.get_electric_field(i, j, h, 'x')
                field_y_grid[i, j] = self.get_electric_field(i, j, h, 'y')
                field_abs_grid[i, j] = np.linalg.norm([field_x_grid[i, j],
                                                       field_y_grid[i, j]])
        if plot:
            self.__plotter.plot_electric_field(field_x_grid,
                                               field_y_grid,
                                               field_abs_grid)
        return field_abs_grid
                
    def get_optimal_relaxation_factor(self, omega_range=None):
        """
        Runs through a given range of relaxation factor and applies the
        SOR method for each one. Returns the relaxation factor which resulted
        in the least iterations.
        @param omega_range: range of relaxation factor to run through
        @return: optimal_relaxation_factor: relaxation factor that results in
                                            the minimum number of iterations.
        """
        if omega_range is None:
            omega_range = self.default_omega_range
        counts = []
        for w in omega_range:
            self.reset_point_dictionary()
            x = self.solve_with_successive_over_relaxation(w)
            counts.append(x)
        optimal_relaxation_factor = omega_range[counts.index(min(counts))]
        return optimal_relaxation_factor
    
    def get_theoretical_relaxation_factor(self):
        """
        Returns a theoretical value of the relaxation factor, which depends on
        the number of unknown points to be solved.
        """
        arg_denom = float(self.__npt_x - self.__ppcm * self.__bar_width - 1)
        eta = (0.5*(np.cos(np.pi/arg_denom) + np.cos(np.pi/arg_denom)))**2
        theoretical_value = 2*(1 - np.sqrt(1 - eta))/eta
        return theoretical_value
        
    def get_potential_profile(self, scaled=False, plot=True):
        """
        Displays the potential profile of the coaxial cable along the
        horizontal axis of symmetry.
        @param scaled: Scale values by the cable width.
        @return:
        """
        potential_profile = []
        if scaled:
            size = int(self.__npt_x)/2 - int(self.__ppcm*self.__bar_width)/2
            sample = np.arange(0, size)
            for i in sample:
                potential_profile.append(self.__point_dict[(i, int(self.__npt_y/2))])
            x_axis = sample/float(self.__ppcm*size - 1)
        else:
            sample = np.arange(0, int(self.__npt_x))
            for i in sample:
                potential_profile.append(self.__point_dict[(i, int(self.__npt_y/2))])
            x_axis = [sample[i]/float(self.__ppcm) - (self.__tube_width/2)
                      for i in range(len(sample))]
        if plot:
            self.__plotter.plot_potential_profile(x_axis,
                                                  potential_profile,
                                                  scaled=scaled)
        return x_axis, potential_profile

    def reset_point_dictionary(self):
        """Resets all values of point dictionary to zero."""
        for i, j in product(range(self.__npt_x), range(self.__npt_y)):
            if not self.__fixed_points[i, j]:
                self.__point_dict[(i, j)] = 0
        
