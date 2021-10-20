import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

from laplace_mesh.laplace_mesh import LaplaceMesh

# Module containing functions which replicate and produce results obtained
# whilst investigating the features of the coaxial cable.

def power_law_fit(x, coeff, exponent):
    """
    Typical power law fitting function used for many components.
    @param x: value
    @param coeff: coefficient of power law
    @param exponent: exponent of power law
    @return:
    """
    power_law = coeff*x**exponent
    return power_law

def optimize_mesh_size(width=16, precision=0.001, max_mesh_density=15):
    """
    Experiment to determine a suitable mesh spacing to reflect the physical
    nature of the coaxial cable problem. The value of the field is expected to 
    converge to the true value as points-per-centimetre is increased. Therefore,
    this function finds the mesh spacing for which the value is found to 
    converge as ppcm is increased.
    """
    points = []
    value = []
    # initial values
    posterior = 200
    mesh_density = 2

    while mesh_density < max_mesh_density:
        laplace_mesh = LaplaceMesh(width, width, ppcm=mesh_density)
        laplace_mesh.solve_with_successive_over_relaxation()
        npx, npy = laplace_mesh.get_n_points()
        new_posterior = laplace_mesh.get_electric_field_grid()[npx//4, npy//2]
        posterior, prior = new_posterior, posterior
        value.append(posterior)
        points.append(mesh_density)
        print("points-per-centimetre = {}".format(mesh_density))
        mesh_density += 1
        if np.abs(float(prior-posterior)/prior) < precision:
            print("Suitable Mesh Spacing = {}".format(mesh_density))
            break

    plt.plot(points, value)
    plt.xlabel("Points-per-centimetre (ppcm)")
    plt.ylabel("Electric field value (V/cm) at x=Lx/4, y=Ly/2")
    plt.grid(b=True, which='both', axis='both', color = 'k')
    plt.show()

def optimize_tolerance(width=16, ppcm=7, precision=0.001,
                       lowest_tol_power=-14, plot=True):
    """This can also be used to show that required tolerance is independent of mesh spacing"""
    tolerances = []
    value = []

    # initial values
    tol_power = -2
    posterior = 200

    while tol_power > lowest_tol_power :
        tol = 10**tol_power
        laplace_mesh = LaplaceMesh(width, width, ppcm=ppcm)
        laplace_mesh.solve_with_successive_over_relaxation(tolerance=tol)
        npx, npy = laplace_mesh.get_n_points()
        new_posterior = laplace_mesh.get_electric_field_grid()[npx//4, npy//2]
        posterior, prior = new_posterior, posterior
        value.append(posterior)
        tolerances.append(tol)
        print("Convergence tolerance = {}".format(tol))
        tol_power -= 1
        tol = 10**tol_power
        if np.abs(float(prior-posterior)/prior) < precision:
            print("Suitable Tolerance = {}".format(tol))
            break 
    if plot:
        plt.loglog(tolerances, value)
        plt.xlabel("Convergence Tolerance")
        plt.ylabel("Electric field value at (x, y) = (Lx/4, Ly/2)")
        plt.grid(b=True, which='both', axis='both', color = 'k')
        plt.show()
    else:
        return 10**(min(tolerances))
    
def iter_vs_ppcm(max_mesh_density=12, width=6, wbar=2, potential=10):
    """
    Experiment B1.1 - Rate of Convergence against Mesh Density:
    Using the jacobi method, iterates to a converged solution for each value of
    points-per-centimetre and records number of iterations required. Also fits
    the data found to a power law curve and plots the idealised curve to the
    plot for comparison.
    """
    mesh_densities = np.arange(1, max_mesh_density)
    iter_count = []
    for mesh_density in mesh_densities:
        laplace_mesh = LaplaceMesh(width, width, wbar=wbar,
                                   ppcm=mesh_density, V=potential)
        res = laplace_mesh.solve_with_jacobi()
        print(mesh_density, res)
        iter_count.append(res)
    results = spo.curve_fit(power_law_fit, mesh_densities, iter_count, (1, 1))
    fit = [results[0][0]*x**results[0][1] for x in mesh_densities]
    plt.plot(mesh_densities, iter_count, 'bo-', linewidth=1.5)
    plt.plot(mesh_densities, fit, 'r--', linewidth=3.5)
    plt.legend(['Data', '{}(ppcm)^{}'.format(round(results[0][0], 2),
                                             round(results[0][1], 2))],
               loc='upper left')
    plt.xlabel("Points-per-cm (ppcm)")
    plt.ylabel("Iterations required")
    plt.grid(b=True, which='both', axis='both', color='k')
    plt.show()
    
def iter_vs_width(min_width=6, max_width=26, wbar=2, ppcm=2, potential=10):
    """
    Experiment B1.2 - Variation of Iterations with Outer Tube Width:
    Uses the Gauss-Seidel method to determine the variation of iteration
    with the width of the outer tube, keeping the inner bar fixed.
    """
    width_values = np.arange(min_width, max_width)
    iterations = []
    for width in width_values:
        laplace_mesh = LaplaceMesh(width, width, wbar=wbar,
                                   ppcm=ppcm, V=potential)
        res = laplace_mesh.solve_with_gauss_seidel()
        print(width, res)
        iterations.append(res)
    results = spo.curve_fit(power_law_fit, width_values, iterations, (1, 1))
    fit = [results[0][0]*x**results[0][1] for x in width_values]
    plt.plot(width_values, iterations, 'bo-', linewidth=1.5)
    plt.plot(width_values, fit, 'r--', linewidth=3.5)
    plt.xlabel("Tube width (cm)")
    plt.ylabel("Iterations until convergence")
    plt.legend(['Data', '{}x^{}'.format(round(results[0][0], 2),
                                        round(results[0][1], 2))],
               loc='upper left')
    plt.grid(b=True, which='both', axis='both', color='k')
    plt.show()

def method_comparison(max_mesh_density=15, width=6, wbar=2, potential=10):
    """
    Experiment B1.3 - Iterations vs. ppcm for Different Algorithms:
    For a 6cm X 6cm system, solves for the potential of the square coaxial cable
    using the following iterative procedures:
        - Jacobi Method
        - Gauss-Seidel Method
        - Successive Over-Relaxation with relaxation factor fixed
        - Successive Over-Relaxation using theoretical optimum relaxation factor
    
    Repeats this for mesh density values of points-per-centimetre ranging from
    1 to 14. For each value of ppcm the number of iterations required to reach
    convergence is recorded (within a fractional tolerance of 1e-8).
    """
    mesh_densities = np.arange(1, max_mesh_density)
    jacobi_count = []
    gauss_seidel_count = []
    sor_count = []
    sor_theory_count = []
    for mesh_density in mesh_densities:
        print(mesh_density)
        laplace_mesh = LaplaceMesh(width, width, wbar=wbar,
                                   ppcm=mesh_density, V=potential)
        jacobi_iter = laplace_mesh.solve_with_jacobi()
        laplace_mesh.reset_point_dictionary()
        gauss_seidel_iter = laplace_mesh.solve_with_gauss_seidel()
        laplace_mesh.reset_point_dictionary()
        sor_iter = laplace_mesh.solve_with_successive_over_relaxation(omega=1.5)
        laplace_mesh.reset_point_dictionary()
        sor_theory_iter = laplace_mesh.solve_with_successive_over_relaxation()

        jacobi_count.append(jacobi_iter)
        gauss_seidel_count.append(gauss_seidel_iter)
        sor_count.append(sor_iter)
        sor_theory_count.append(sor_theory_iter)

    plt.plot(mesh_densities, jacobi_count, 'g')
    plt.plot(mesh_densities, gauss_seidel_count, 'b')
    plt.plot(mesh_densities, sor_count, 'r')
    plt.plot(mesh_densities, sor_theory_count, 'm')
    plt.legend(['Jacobi', 'Gauss Seidel', 'SOR W=1.5', 'SOR (~optimal W)'], loc='upper left')
    plt.xlabel("Points-per-centimetre (ppcm)")
    plt.ylabel("Iterations required")
    plt.grid(b=True, which='both', axis='both', color='k')
    plt.show()
    
def optimum_relaxation(width=6):
    """
    Uses the Successive Over-Relaxation Method to solve for the potential of
    a 6cm X 6cm square coaxial cable for a variety of mesh point densities.
    At each value of points per centimetre, estimates optimum relaxation factor
    and plots, to determine relationship.
    """
    mesh_densities = np.arange(1, 8)
    optimal_factor_list = []
    theory_factor_list = []
    for mesh_density in mesh_densities:
        laplace_mesh = LaplaceMesh(width, width, ppcm=mesh_density)
        optimal_factor = laplace_mesh.get_optimal_relaxation_factor()
        theory_factor = laplace_mesh.get_theoretical_relaxation_factor()
        print(mesh_density, optimal_factor, theory_factor)
        optimal_factor_list.append(optimal_factor)
        theory_factor_list.append(theory_factor)
    plt.plot(mesh_densities, optimal_factor_list)
    plt.plot(mesh_densities, theory_factor_list)
    plt.legend(['Numerical', 'Theoretical'], loc='lower right')
    plt.xlabel("Points-per-centimetre")
    plt.ylabel("Optimal Relaxation Factor")
    plt.grid(b=True, which='both', axis='both', color='k')
    plt.show()
    
def tube_variation(ppcm=2):
    """
    Plots a scaled potential profile of one side of the square coaxial cable
    showing the variation as the distance of the outer tube from the inner
    bar is varied.
    """
    for wtube in range(4, 80, 4):
        laplace_mesh = LaplaceMesh(wtube, wtube, ppcm=ppcm)
        laplace_mesh.solve_with_successive_over_relaxation()
        plt.figure(1)
        laplace_mesh.get_potential_profile(scaled=True)
        plt.xlabel("Relative distance from tube to bar (cm)")
        plt.ylabel("Potential (V)")
        plt.grid(b=True, which='both', axis='both', color='k')
        plt.show()
    
def profile_ensemble(sizes=[4, 35]):
    """
    For wtube ~ wbar and wtube >> wbar, the potential profile along the centre
    axis of the cross-section for the coaxial cable is plotted.
    Additionally, 3D surfaces are found that plot the "potential landscape" of
    the physical problem.
    """
    for i in range(len(sizes)):
        laplace_mesh = LaplaceMesh(sizes[i], sizes[i], ppcm=7)
        laplace_mesh.solve_with_successive_over_relaxation()
        plt.subplot(1, 2, i+1)
        laplace_mesh.get_potential_profile()
        plt.title("Tube Width = {} cm".format(float(sizes[i])))
    plt.show()

        

        
    