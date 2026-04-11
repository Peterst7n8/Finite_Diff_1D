from finite_diff_1d import Material, Solver, extrapolated_distance
import scipy.interpolate as interp
import time
import numpy as np
import matplotlib.pyplot as plt

"""
Author : Pierre Boussemart

This script demonstrates how to use of the package finite_diff_1d and 
highlights the accuracy of the diffusion solver compared to Monte Carlo reference solutions.

This script demonstrates :
1. How to create materials and import cross-sections,
2. How to build a geometry and use it in a calculation
3. Solve the multigroup diffusion equation for the geometry input
4. Compares the results with a reference calculation's results (OpenMC)

Physics : 
- Simple 1D heterogenous geometry : cf. fuel_slab_water_reflector.png
- Multigroup neutron diffusion equation 
- Vacuum boundary condition on the left and right

Input : 
- Geometry and boundary conditions
- 4-Group Macroscopic Cross Sections and Diffusion Coefficients
- Spatial discretization Step

Ouput : 
- Thermal (<= 0.625 eV) and Fast (>0.625 eV) Flux Distributions
- Computation time
- Eigenvalue associated with system
- Absolute error on eigenvalue (OpenMC reference calculation)
- Relative Error on flux (OpenMC reference calculation)

"""


## --------------- MATERIALS DEFINITION ---------------- ##

start = time.time()

# We start by defining the fuel composition, even though there is no use to it (macroscopic cross sections
# are given to the code hence no conversion from microscopic to macroscopic is needed)

fuel_comp = {"U238": 0.8, "U235": 0.2, "O16": 2}

# We define its density even though there is no need (cf. previous paragraph)
fuel_density = 10.5


# We create a material instance, assign it the defined composition and density, and set the
# group number to 4.
fuel = Material(nuclides=fuel_comp, macro=True, density=fuel_density, groups=4)

# We import the multigroup scattering matrix
fuel.get_macro_xs("scat", "./examples/mgxs_water_void/xs_scat_core.csv")
# We import the multigroup fission matrix
fuel.get_macro_xs("fiss", "./examples/mgxs_water_void/xs_f_core.csv")
# We import the multigroup absorption cross section
fuel.get_macro_xs("abs", "./examples/mgxs_water_void/xs_a_core.csv")
# We import the multigroup diffusion coefficient
fuel.get_diff("./examples/mgxs_water_void/diff_core.csv")

# We repeat the operation for the water Material :
# - definition of composition,
# - definition of density,
# - definition of number of groups

refl_comp = {"H0": 2, "O0": 1}
refl_density = 1

refl = Material(nuclides=refl_comp, macro=True, density=refl_density, groups=4)

# We import the multigroup scattering matrix
refl.get_macro_xs("scat", "./examples/mgxs_water_void/xs_scat_refl.csv")
# We import the multigroup absorption cross section
refl.get_macro_xs("abs", "./examples/mgxs_water_void/xs_a_refl.csv")
# We import the multigroup diffusion coefficient
refl.get_diff("./examples/mgxs_water_void/diff_refl.csv")

t_xs = time.time() - start
start = time.time()


## --------------- GEOMETRY DEFINITION ---------------- ##

# Now that the materials have been created, we can start the creation of our geometry
# and the definition of our boundary conditions

# WARNING :
# A vacuum boundary condition in neutronics doesn't translate to a zero
# flux condition at the end of the domain (here, at the edge of the water reflectors),
# but into a zero flux condition at the extrapolated distance d:
# where d = 0.7104/Transport macroscopic cross section = 0.7104 * 3 * D_g
# Hence we had the distance d to the first and last regions of the domain

# We precise the discretization step
h = 0.1

# We calculate the extrapolated distance using the function extrapolated_distance
# defined in Solver.py
extra_dist = extrapolated_distance(refl, h, False)

# We create the geometry
geom = {refl: 20 + extra_dist, fuel: 20, refl.clone(): 20 + extra_dist}

# We define the boundary conditions
bc_right = "void"
bc_left = "void"

# We define the spatial discretization step for each region

# We create an instance of Solver that will contain the results, matrices, etc...
f_1d = Solver(groups=4, geom=geom, step=[h, h, h], bc_left=bc_left, bc_right=bc_right)


## --------------- CALCULATION ---------------- ##


# We ask the solver to prepare the matrices for the calculation
# WARNING : when h<0.01, the time for the assembly of the matrices can be long

t_mat = time.time() - start
start = time.time()

# We launch the calculation
f_1d.compute(1000, 1, 1e-7, 1)

t_solve = time.time() - start


# We announce the resulting eigenvalue
print("\nk-eff :", f_1d.k)

# We have the reference eigenvalue and compute the absolute error between the result
# of the calculation and the reference eigenvalue
k_ref = 1.08232
print("Err k-eff", (f_1d.k - k_ref) * 1e5)


## --------------- POST-PROCESSING ---------------- ##


# We compute the thermal and fast flux depending the number of groups.
# The groups structures are CASMO-X, with X = {4,8,16,25,70}

flux_thermique = np.zeros(shape=(f_1d.nb_nodes,), dtype=float)
flux_rapide = np.zeros(shape=(f_1d.nb_nodes,), dtype=float)

# We split energy groups into fast and thermal ranges (CASMO structure)
for g in range(f_1d.groups):
    if f_1d.groups == 4:
        if g >= 3:
            flux_thermique += f_1d.phi[g * f_1d.nb_nodes : (g + 1) * f_1d.nb_nodes]
        else:
            flux_rapide += f_1d.phi[g * f_1d.nb_nodes : (g + 1) * f_1d.nb_nodes]
    elif f_1d.groups == 16 or f_1d.groups == 8:
        if g >= 3:
            flux_thermique += f_1d.phi[g * f_1d.nb_nodes : (g + 1) * f_1d.nb_nodes]
        else:
            flux_rapide += f_1d.phi[g * f_1d.nb_nodes : (g + 1) * f_1d.nb_nodes]
    elif f_1d.groups == 25:
        if g >= 13:
            flux_thermique += f_1d.phi[g * f_1d.nb_nodes : (g + 1) * f_1d.nb_nodes]
        else:
            flux_rapide += f_1d.phi[g * f_1d.nb_nodes : (g + 1) * f_1d.nb_nodes]
    elif f_1d.groups == 70:
        if g >= 27:
            flux_thermique += f_1d.phi[g * f_1d.nb_nodes : (g + 1) * f_1d.nb_nodes]
        else:
            flux_rapide += f_1d.phi[g * f_1d.nb_nodes : (g + 1) * f_1d.nb_nodes]


# We normalize the computed thermal and fast fluxes
flux_thermique = flux_thermique / flux_thermique.max()
flux_rapide = flux_rapide / flux_rapide.max()

# We load the reference thermal and fast flux generated via OpenMC
flux_mc = np.loadtxt("./examples/flux_mc_w_void.csv", dtype=float)

# We normalize it too
flux_th_mc = flux_mc[:, 0] / flux_mc[:, 0].max()
flux_f_mc = flux_mc[:, 1] / flux_mc[:, 1].max()

# We recreate the grid the reference flux was created on
x_mc_init = np.linspace(-30, 30, 200)

# We create the grid used by the solver for its calculation
x = np.arange(-30 - extra_dist, 30 + extra_dist + h, h)

# We interpolate the reference flux with the calculation grid and ask for an extrapolation of the
# reference flux onto the extrapolated_distance
interp_th = interp.interp1d(x_mc_init, flux_th_mc, copy=True, fill_value="extrapolate")
interp_f = interp.interp1d(x_mc_init, flux_f_mc, copy=True, fill_value="extrapolate")
flux_th_mc = interp_th(x)
flux_f_mc = interp_f(x)

# We calculate the energy integrated flux for the reference calculation and the solver calculation
flux_tot_mc = flux_th_mc + flux_f_mc
flux_tot = flux_thermique + flux_rapide

# We compute the relative error of the solver on the energy integrated flux
err = (flux_tot / flux_tot_mc) - 1

# We inform the user of the maximum relative error (in absolute value)
# WARNING : As the reference flux has been extrapolated to it the grid of the
# solver's calculation (the additionnal distance for the zero flux condition),
# we do not consider the error in the extrapolated distance.
if int(extra_dist / h) != 0:
    err_max = np.max(np.abs(err[int(extra_dist / h) + 1 : -int(extra_dist / h) - 1])) * 100
else:
    err_max = np.max(np.abs(err)) * 100
print("Maximum Relative Error (Absolute value) :", err_max, "%")


# We plot the thermal and fast flux distributions obtained via the solver
plt.plot(x, (flux_thermique / flux_thermique.max()), label="Thermal flux Approx")
plt.plot(x, (flux_rapide / flux_rapide.max()), label="Fast Flux Approx")

# We plot the reference's thermal and fast flux distributions
plt.plot(x, flux_th_mc, label="Thermal Flux OpenMC")
plt.plot(x, flux_f_mc, label="Fast Flux OpenMC")

# We label and limit the graph to the region of interest
plt.xlabel("Position (cm)")
plt.ylabel("Flux intensity (A.U)")
plt.title("Comparison of the flux obtained with the solver\n and the exact flux from OpenMC")
plt.plot(x, err, label="Relative Error on the Energy Integrated Neutron Flux")


plt.legend(loc="upper right")
plt.show()

# Final print

print(
    f"""
--- Simulation Summary ---
k-eff                 : {f_1d.k:.5f}
Error k-eff (pcm)     : {(f_1d.k - k_ref)*1e5:.1f}
Max. Rel. Error Flux  : {err_max:.2f} %
Time                  : {t_xs+t_mat+t_solve:.2f} s
"""
)
