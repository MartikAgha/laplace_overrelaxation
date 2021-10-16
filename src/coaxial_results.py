import numpy as np
import matplotlib.pyplot as plt
from laplace_mesh import *
import scipy.optimize as spo
"""
Module containing functions which replicate and produce results obtained whilst
investigating the features of the coaxial cable.
"""

def fit_func(x, A, b):
    """Typical power law fitting function used for many components."""
    return A*x**b

def suitable_mesh(width=16, precision=0.001):
    """
    Experiment to determine a suitable mesh spacing to reflect the physical
    nature of the coaxial cable problem. The value of the field is expected to 
    converge to the true value as points-per-centimetre is increased. Therefore,
    this function finds the mesh spacing for which the value is found to 
    converge as ppcm is increased.
    """
    points = []
    value = []
    p=2
    posterior = 200
    while p < 15:
        M = LaplaceMesh(width, width, ppcm=p)
        M.solve_with_successive_over_relaxation()
        posterior, prior = M.get_electric_field_grid()[M.__npoints_x / 4, M.__n_points_y / 2.], posterior
        value.append(posterior)
        points.append(p)
        print("points-per-centimetre = {}".format(p))
        p += 1
        if np.abs(float(prior-posterior)/prior) < precision:
            print("Suitable Mesh Spacing = {}".format(p))
            break 
    plt.plot(points, value)
    plt.xlabel("Points-per-centimetre (ppcm)")
    plt.ylabel("Electric field value (V/cm) at x=Lx/4, y=Ly/2")
    plt.grid(b=True, which='both', axis='both', color = 'k')
    plt.show()

def suitable_tol(width=16, ppcm=7, precision=0.001, plot=True):
    """This can also be used to show that required tolerance is independent of mesh spacing"""
    tolerances = []
    value = []
    tol= -2
    posterior = 200
    while tol > -14 :
        M = LaplaceMesh(width, width, ppcm=ppcm)
        M.solve_with_successive_over_relaxation(tolerance=10 ** (tol))
        posterior, prior = M.get_electric_field_grid()[M.__npoints_x / 4., M.__n_points_y / 2.], posterior
        value.append(posterior)
        tolerances.append(10**(tol))
        print("Convergence tolerance = {}".format(10**(tol)))
        tol -= 1
        if np.abs(float(prior-posterior)/prior) < precision:
            print("Suitable Tolerance = {}".format(10**(tol)))
            break 
    if plot == True:
        plt.loglog(tolerances, value)
        plt.xlabel("Convergence Tolerance")
        plt.ylabel("Electric field value at (x, y) = (Lx/4, Ly/2)")
        plt.grid(b=True, which='both', axis='both', color = 'k')
        plt.show()
    else:
        return 10**(min(tolerances))
    
def iter_vs_ppcm(maximum=12):
    """
    Experiment B1.1 - Rate of Convergence against Mesh Density:
    Using the jacobi method, iterates to a converged solution for each value of
    points-per-centimetre and records number of iterations required. Also fits
    the data found to a power law curve and plots the idealised curve to the
    plot for comparison.
    """
    point_density = np.arange(1, maximum)
    iter_count = []
    for P in point_density:
        M = LaplaceMesh(6, 6, wbar = 2, ppcm = P, V = 10)
        res = M.solve_with_jacobi()
        print (P, res)
        iter_count.append(res)
    results = spo.curve_fit(fit_func, point_density, iter_count, (1, 1))
    fit = [results[0][0]*x**results[0][1] for x in point_density]
    plt.plot(point_density, iter_count, 'bo-', linewidth=1.5)
    plt.plot(point_density, fit, 'r--', linewidth=3.5)
    plt.legend(['Data', '{}(ppcm)^{}'.format(round(results[0][0], 2), round(results[0][1], 2))], loc='upper left')
    plt.xlabel("Points-per-cm (ppcm)")
    plt.ylabel("Iterations required")
    plt.grid(b=True, which='both', axis='both', color='k')
    plt.show()
    
def iter_vs_width(maximum=26):
    """
    Experiment B1.2 - Variation of Iterations with Outer Tube Width:
    Uses the Gauss-Seidel method to determine the variation of iteration
    with the width of the outer tube, keeping the inner bar fixed.
    """
    width = np.arange(6, maximum)
    iterations = []
    for Wtube in width:
        M = LaplaceMesh(Wtube, Wtube, wbar=2, ppcm=2, V=10)
        res = M.solve_with_gauss_seidel()
        print (Wtube, res)
        iterations.append(res)
    results = spo.curve_fit(fit_func, width, iterations, (1, 1))
    fit = [results[0][0]*x**results[0][1] for x in width]
    plt.plot(width, iterations, 'bo-', linewidth=1.5)
    plt.plot(width, fit, 'r--', linewidth=3.5)
    plt.xlabel("Tube width (cm)")
    plt.ylabel("Iterations until convergence")
    plt.legend(['Data', '{}x^{}'.format(round(results[0][0], 2), round(results[0][1], 2))], loc='upper left')
    plt.grid(b=True, which='both', axis='both', color='k')
    plt.show()

def method_comparison():
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
    point_density = np.arange(1, 15)
    Jacobi_count = []
    GS_count = []
    SOR_count = []
    Theory_factor = []
    for P in point_density:
        print (P)
        M = LaplaceMesh(6, 6, wbar = 2, ppcm = P, V = 10)
        res = M.solve_with_jacobi()
        Jacobi_count.append(res)
        M = LaplaceMesh(6, 6, wbar = 2, ppcm = P, V = 10)
        res = M.solve_with_gauss_seidel()
        GS_count.append(res)
        M = LaplaceMesh(6, 6, wbar = 2, ppcm = P, V = 10)
        res = M.solve_with_successive_over_relaxation(omega= 1.5)
        SOR_count.append(res)
        M = LaplaceMesh(6, 6, wbar = 2, ppcm = P, V = 10)
        res = M.solve_with_successive_over_relaxation()
        Theory_factor.append(res)
    plt.plot(point_density, Jacobi_count, 'g')
    plt.plot(point_density, GS_count, 'b')
    plt.plot(point_density, SOR_count, 'r')
    plt.plot(point_density, Theory_factor, 'm')
    plt.legend(['Jacobi', 'Gauss Seidel', 'SOR W=1.5', 'SOR (~optimal W)'], loc='upper left')
    plt.xlabel("Points-per-centimetre (ppcm)")
    plt.ylabel("Iterations required")
    plt.grid(b=True, which='both', axis='both', color='k')
    plt.show()
    
def optimum_relaxation():
    """
    Uses the Successive Over-Relaxation Method to solve for the potential of
    a 6cm X 6cm square coaxial cable for a variety of mesh point densities.
    At each value of points per centimetre, estimates optimum relaxation factor
    and plots, to determine relationship.
    """
    P_count = np.arange(1, 8)
    Opt = []
    theory = []
    for p in P_count:
        M = LaplaceMesh(6, 6, ppcm=p)
        W = M.get_optimal_relaxation_factor()
        T = M.get_theoretical_relaxation_factor()
        print (p, W, T)
        Opt.append(W)
        theory.append(T)
    plt.plot(P_count, Opt)
    plt.plot(P_count, theory)
    plt.legend(['Numerical', 'Theoretical'], loc='lower right')
    plt.xlabel("Points-per-centimetre")
    plt.ylabel("Optimal Relaxation Factor")
    plt.grid(b=True, which='both', axis='both', color='k')
    plt.show()
    
def tube_variation():
    """
    Plots a scaled potential profile of one side of the square coaxial cable
    showing the variation as the distance of the outer tube from the inner
    bar is varied.
    """
    for wtube in range(4, 80, 4):
        M = LaplaceMesh(Lx=wtube, Ly=wtube, ppcm = 2)
        M.solve_with_successive_over_relaxation()
        plt.figure(1)
        M.profile(scaled = True)
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
        M = LaplaceMesh(sizes[i], sizes[i], ppcm=7)
        M.solve_with_successive_over_relaxation()
        plt.subplot(1, 2, i+1)
        M.profile()
        plt.title("Tube Width = {} cm".format(float(sizes[i])))
    plt.show()

        

        
    