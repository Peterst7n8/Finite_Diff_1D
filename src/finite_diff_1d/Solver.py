from .Materials import *
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def extrapolated_distance(mat: Material, h: float, over: bool = True):
    """
    Compute an extrapolated distance adjusted to the discretization step.

    This function adjusts a physical distance so that it matches a multiple
    of the discretization step `h`. A correction factor is applied when
    the mesh is fine (h < 0.5).

    Parameters
    ----------
    distance : float
        Physical distance to adjust.
    h : float
        Spatial discretization step.
    over : bool, optional
        If True, rounds up to the next multiple of `h`.
        If False, rounds down. Default is True.

    Returns
    -------
    float
        Adjusted distance compatible with the discretization.
    """

    distance = np.min(3 * mat._diff_coef)

    if h >= 0.5:
        if over:
            return (int(distance / h) + 1) * h
        else:
            return int(distance / h) * h
    else:
        if over:
            return (int((0.7104 * distance) / h) + 1) * h
        else:
            return (int((0.7104 * distance) / h)) * h


class Solver:
    """
    Finite difference solver for multigroup neutron diffusion problems.

    This class builds and solves the discretized neutron diffusion equations
    over a 1D or 2D domain composed of multiple materials.

    The domain is divided into regions of constant material properties,
    each discretized with its own spatial step.

    Parameters
    ----------
    groups : int, optional
        Number of energy groups. Default is 1.
    dim : int, optional
        Problem dimension (1 or 2). Default is 1.
    geom : dict[Material, float], optional
        Dictionary mapping each material to its physical size.
        The order defines the geometry. Default is a single region.
    step : array-like, optional
        Spatial discretization step for each region.
    method : str, optional
        Discretization scheme ('Pierre', 'A-H', etc.). Default is 'Pierre'.
    bc_left : str, optional
        Left boundary condition ('void', 'reflective'). Default is 'void'.
    bc_right : str, optional
        Right boundary condition ('void', 'reflective'). Default is 'void'.

    Attributes
    ----------
    phi : ndarray
        Neutron flux vector.
    A : scipy.sparse matrix
        Diffusion operator matrix.
    S : scipy.sparse matrix
        Scattering matrix.
    F : scipy.sparse matrix
        Fission matrix.
    k : float
        Effective multiplication factor.
    """

    def __init__(
        self,
        groups: int = 1,
        dim: int = 1,
        geom: dict[Material, float] = {Material(nuclides={"U238": 1}, macro=False, density=1, groups=1): 1},
        step: np.ndarray = [1],
        method: str = "Pierre",
        bc_left: str = "void",
        bc_right: str = "void",
        plot: bool = False,
        edge: bool = False,
    ) -> "Solver":
        """
        groups : int, number of groups in calculation
        geom : dict[Material,size], order of Material and size determines Geometry
        step : list of float, discretization step of each Material
        """

        self.groups = groups
        i = 0
        self.materials = np.zeros(len(geom), Material)
        self.region_size = np.zeros(len(geom))
        for mat, val in geom.items():
            self.materials[i] = mat
            self.region_size[i] = val
            i += 1
        self.step = step

        self.nb_nodes = int(0)
        self.nb_nodes_region = np.zeros(len(geom), int)
        if len(self.materials) != len(self.step):
            if len(self.materials) >= len(self.step):
                raise ValueError("Invalid input : not enough discretization steps or " "too many materials")
            else:
                raise ValueError("Invalid input : too many discretization steps or too few " "Materials")

        if len(self.materials) != 1:
            for i in range(len(self.materials)):
                reste = self.region_size[i] - step[i] * round(self.region_size[i] / step[i])
                tol = 1e-9
                if reste <= tol:
                    self.nb_nodes += int(self.region_size[i] / self.step[i])
                    if i == 0 or i == len(self.materials) - 1:
                        self.nb_nodes_region[i] = self.region_size[i] / self.step[i]
                    else:
                        self.nb_nodes_region[i] = (self.region_size[i] / self.step[i]) - 1
                else:
                    print(self.region_size[i], step[i])
                    print(self.region_size[i] // step[i], self.region_size[i] % step[i])
                    raise ValueError("Discretization step incompatible with domain sizes")
            self.nb_nodes += 1
        else:
            self.nb_nodes = int(self.region_size[0] / self.step[0]) + 1
            self.nb_nodes_region = [self.nb_nodes]

        self.method = method
        self.bc_right = bc_right
        self.bc_left = bc_left

        self._prep_matrixes(plot=plot, edge=edge)

    def _prep_matrixes(self, plot, edge) -> None:
        """
        Assemble the global matrices A, S, and F for all energy groups.

        Each matrix is built block-wise from spatial sub-matrices.

        Parameters
        ----------
        plot : bool
            If True, display matrix heatmaps for debugging/visualization.
        edge : bool
            If True, includes edge contributions (boundary weighting).
        """

        active_groups = []

        for g in range(self.groups):
            is_active = False

            for mat in self.materials:

                if mat._a_xs[g] > 0 or np.any(mat._nu_f_xs[g, :]) or np.any(mat._scat_xs[g, :]):
                    is_active = True
                    break

            if is_active:
                active_groups.append(g)

        if len(active_groups) == 0:
            raise ValueError("No active energy groups found.")

        self.active_groups = active_groups
        self.G_eff = len(active_groups)

        N = self.nb_nodes
        G = self.G_eff

        self.A = sp.lil_matrix((N * G, N * G), dtype=float)
        self.S = sp.lil_matrix((N * G, N * G), dtype=float)
        self.F = sp.lil_matrix((N * G, N * G), dtype=float)

        for i in range(G):
            self.A[i * self.nb_nodes : (i + 1) * self.nb_nodes, i * self.nb_nodes : (i + 1) * self.nb_nodes] = self.spatial_matrix_A(i, edge)
        for g_out in range(G):
            for g_in in range(G):
                self.S[g_in * self.nb_nodes : (g_in + 1) * self.nb_nodes, g_out * self.nb_nodes : (g_out + 1) * self.nb_nodes] = (
                    self.spatial_matrix_S(g_out, g_in, edge)
                )
                self.F[g_in * self.nb_nodes : (g_in + 1) * self.nb_nodes, g_out * self.nb_nodes : (g_out + 1) * self.nb_nodes] = (
                    self.spatial_matrix_F(g_out, g_in, edge)
                )

        if plot:
            ax = plt.matshow(self.S.toarray(), norm=colors.LogNorm())
            plt.colorbar(ax)
            plt.title("Matrix S")
            plt.show()

            ax = plt.matshow(self.F.toarray(), norm=colors.LogNorm())
            plt.colorbar(ax)
            plt.title("Matrix F")
            plt.show()

            ax = plt.matshow(self.A.toarray(), norm=colors.LogNorm())
            plt.colorbar(ax)
            plt.title("Matrix A")
            plt.show()

    def A_B_Pi(self, group, edge) -> sp.spmatrix:
        """
        Build the diffusion matrix A for a given energy group using
        the standard finite difference formulation.

        The matrix is tridiagonal and accounts for:
        - diffusion
        - absorption
        - scattering
        - boundary conditions

        Special treatment is applied at material interfaces using
        harmonic averaging of diffusion coefficients.

        Parameters
        ----------
        group : int
            Energy group index.
        plot : bool
            Unused (kept for interface consistency).
        edge : bool
            If True, modifies boundary contributions.

        Returns
        -------
        scipy.sparse matrix
            Tridiagonal diffusion matrix for the group.
        """

        A_mid = np.zeros((int(self.nb_nodes)), dtype=float)
        A_up = np.zeros((int(self.nb_nodes) - 1), dtype=float)
        A_down = np.zeros((int(self.nb_nodes) - 1), dtype=float)

        done = 0
        for i in range(len(self.materials)):
            a = (
                (2 * (self.materials[i]._diff_coef[group]) / self.step[i] ** 2)
                + self.materials[i]._a_xs[group]
                + np.sum(self.materials[i]._scat_xs[group, :])
            )
            D = -(self.materials[i]._diff_coef[group]) / (self.step[i] ** 2)

            if self.nb_nodes_region[i] == 0:
                j = 0
                D_left = self.materials[i]._diff_coef[group]
                D_right = self.materials[i + 1]._diff_coef[group]

                A_up[done + j] = -D_right / (self.step[i + 1])
                A_mid[done + j] = (
                    D_left / self.step[i]
                    + D_right / self.step[i + 1]
                    + 1
                    / 2
                    * (
                        self.step[i] * (self.materials[i]._a_xs[group] + np.sum(self.materials[i]._scat_xs[group, :]))
                        + self.step[i + 1] * (self.materials[i + 1]._a_xs[group] + np.sum(self.materials[i + 1]._scat_xs[group, :]))
                    )
                )
                A_down[done + j - 1] = -D_left / (self.step[i])

                done += 1
                continue

            for j in range(self.nb_nodes_region[i]):
                if i == 0 and j == 0:
                    match self.bc_left:
                        case "reflective":
                            A_up[done + j] = -(self.materials[i]._diff_coef[group]) / (self.step[i])
                            if edge:
                                A_mid[done + j] = (
                                    (self.materials[i]._diff_coef[group]) / (self.step[i])
                                    + self.materials[i]._a_xs[group]
                                    + np.sum(self.materials[i]._scat_xs[group, :])
                                )
                            else:
                                A_mid[done + j] = (self.materials[i]._diff_coef[group]) / (self.step[i])
                        case "void":
                            A_up[done + j] = D
                        case default:
                            raise ValueError("Unrecognised BC type for left BC")
                elif i == len(self.materials) - 1 and j == self.nb_nodes_region[i] - 1:
                    match self.bc_right:
                        case "reflective":
                            A_down[-1] = -(self.materials[i]._diff_coef[group]) / (self.step[i])
                            if edge:
                                A_mid[done + j] = (
                                    (self.materials[i]._diff_coef[group]) / (self.step[i])
                                    + self.materials[i]._a_xs[group]
                                    + np.sum(self.materials[i]._scat_xs[group, :])
                                )
                            else:
                                A_mid[done + j] = (self.materials[i]._diff_coef[group]) / (self.step[i])
                        case "void":
                            A_down[-1] = D
                        case default:
                            raise ValueError("Unrecognised BC type for right BC")
                else:
                    A_up[done + j] = D
                    A_mid[done + j] = a
                    A_down[done + j - 1] = D
                if i != len(self.materials) - 1 and j == self.nb_nodes_region[i] - 1:

                    j += 1
                    D_left = self.materials[i]._diff_coef[group]
                    D_right = self.materials[i + 1]._diff_coef[group]

                    A_up[done + j] = -D_right / (self.step[i + 1])
                    A_mid[done + j] = (
                        D_left / self.step[i]
                        + D_right / self.step[i + 1]
                        + 1
                        / 2
                        * (
                            self.step[i] * (self.materials[i]._a_xs[group] + np.sum(self.materials[i]._scat_xs[group, :]))
                            + self.step[i + 1] * (self.materials[i + 1]._a_xs[group] + np.sum(self.materials[i + 1]._scat_xs[group, :]))
                        )
                    )
                    A_down[done + j - 1] = -D_left / (self.step[i])
            done += j + 1

        A = sp.diags([A_down, A_mid, A_up], [-1, 0, 1])

        return A

    def spatial_matrix_A(self, group, edge) -> sp.spmatrix:
        """
        Select and build the diffusion matrix A for a given group
        depending on the chosen numerical method.

        Parameters
        ----------
        group : int
            Energy group index.
        plot : bool
            Visualization flag.
        edge : bool
            Boundary treatment flag.

        Returns
        -------
        scipy.sparse matrix
            Diffusion matrix for the group.
        """

        match self.method:
            case default:
                A = self.A_B_Pi(group, edge)
        return A

    def spatial_matrix_S(self, group_in, group_out, edge) -> sp.spmatrix:
        """
        Build the scattering matrix S between two energy groups.

        This matrix represents neutron scattering from group_out
        to group_in.

        Parameters
        ----------
        group_in : int
            Incoming energy group.
        group_out : int
            Outgoing energy group.
        plot : bool
            Unused (kept for interface consistency).
        edge : bool
            If True, includes boundary weighting.

        Returns
        -------
        scipy.sparse matrix
            Scattering matrix.
        """

        S_up = np.zeros((self.nb_nodes - 1), float)
        S_mid = np.zeros((self.nb_nodes), float)
        S_down = np.zeros((self.nb_nodes - 1), float)

        done = 0
        for i in range(len(self.materials)):

            s = self.materials[i]._scat_xs[group_in, group_out]

            if self.nb_nodes_region[i] == 0:
                j = 0

                S_up[done + j] = 1 / 2 * self.step[i + 1] * self.materials[i + 1]._scat_xs[group_in, group_out]
                S_down[done + j - 1] = 1 / 2 * self.step[i] * self.materials[i]._scat_xs[group_in, group_out]

                done += 1
                continue

            for j in range(self.nb_nodes_region[i]):

                if (i == 0 and j == 0) and not (edge):
                    continue
                elif (i == len(self.materials) - 1 and j == self.nb_nodes_region[i] - 1) and not (edge):
                    continue
                elif (i == 0 and j == 0) and edge:
                    S_mid[done + j] = 1 / 2 * self.step[i] * self.materials[i]._scat_xs[group_in, group_out]
                    continue
                elif (i == len(self.materials) - 1 and j == self.nb_nodes_region[i] - 1) and edge:
                    S_mid[done + j] = 1 / 2 * self.step[i] * self.materials[i]._scat_xs[group_in, group_out]
                    continue

                else:
                    S_mid[done + j] = s

                if i != len(self.materials) - 1 and j == self.nb_nodes_region[i] - 1:

                    S_up[done + j] = 1 / 2 * self.step[i] * self.materials[i]._scat_xs[group_in, group_out]

                    j += 1

                    S_mid[done + j] = 0
                    S_down[done + j] = 1 / 2 * self.step[i + 1] * self.materials[i + 1]._scat_xs[group_in, group_out]

            done += j + 1

        S = sp.diags([S_down, S_mid, S_up], [-1, 0, 1])

        return S

    def spatial_matrix_F(self, group_in, group_out, edge) -> sp.spmatrix:
        """
        Build the fission production matrix F.

        This matrix represents neutron production due to fission,
        including spatial weighting and group coupling.

        Parameters
        ----------
        group_in : int
            Produced neutron energy group.
        group_out : int
            Parent neutron energy group.
        plot : bool
            Unused (kept for interface consistency).
        edge : bool
            If True, includes boundary weighting.

        Returns
        -------
        scipy.sparse matrix
            Fission matrix.
        """

        F_mid = np.zeros((self.nb_nodes), dtype=float)
        F_up = np.zeros((self.nb_nodes - 1), dtype=float)
        F_down = np.zeros((self.nb_nodes - 1), dtype=float)

        done = 0
        for i in range(len(self.materials)):

            f = self.materials[i]._nu_f_xs[group_in, group_out]

            if self.nb_nodes_region[i] == 0:
                j = 0

                F_mid[done + j] = 0
                F_up[done + j] = 1 / 2 * self.step[i + 1] * self.materials[i + 1]._nu_f_xs[group_in, group_out]
                F_down[done + j - 1] = 1 / 2 * self.step[i] * self.materials[i]._nu_f_xs[group_in, group_out]

                done += 1
                continue

            for j in range(self.nb_nodes_region[i]):

                if ((i == 0 and j == 0) or (i == len(self.materials) - 1 and j == self.nb_nodes_region[i] - 1)) and not (edge):
                    continue
                elif (i == 0 and j == 0) and edge:
                    F_mid[done + j] = 1 / 2 * self.materials[i]._nu_f_xs[group_in, group_out]
                elif (i == len(self.materials) - 1 and j == self.nb_nodes_region[i] - 1) and edge:
                    F_mid[done + j] = 1 / 2 * self.materials[i]._nu_f_xs[group_in, group_out]
                elif self.method == "A-H":
                    F_up[done + j] = 1 / 2 * self.materials[i]._nu_f_xs[group_in, group_out]
                    F_down[done + j - 1] = 1 / 2 * self.materials[i]._nu_f_xs[group_in, group_out]
                else:
                    F_mid[done + j] = f

                if i != len(self.materials) - 1 and j == self.nb_nodes_region[i] - 1:
                    F_up[done + j] = 1 / 2 * self.step[i] * self.materials[i]._nu_f_xs[group_in, group_out]

                    j += 1

                    F_mid[done + j] = 0
                    F_down[done + j] = 1 / 2 * self.step[i + 1] * self.materials[i + 1]._nu_f_xs[group_in, group_out]

            done += j + 1

        F = sp.diags([F_down, F_mid, F_up], [-1, 0, 1])

        return F

    def compute(self, itext, itint, eps, relax: float) -> None:
        """
        Solve the neutron diffusion eigenvalue problem using
        a power iteration scheme with relaxation.

        The method iteratively updates:
        - the neutron flux (phi)
        - the multiplication factor (k)

        until convergence is reached.

        Parameters
        ----------
        itext : int
            Maximum number of outer iterations.
        itint : int
            Unused (reserved for inner iterations).
        eps : float
            Convergence tolerance for both flux and k.
        relax : float
            Relaxation factor (0 < relax <= 1).

        Returns
        -------
        None

        Notes
        -----
        The final flux is normalized such that its sum equals 1.
        The multiplication factor is stored in `self.k`.
        """
        self.phi = np.ones((self.G_eff * self.nb_nodes, 1), float)
        k = 1
        Q = (self.S + self.F) @ self.phi
        phinew = np.zeros((self.G_eff * self.nb_nodes, 1), float)
        err_k = float(0)
        err_q = float(0)
        for i in range(itext):
            phinew = spla.spsolve(self.A.tocsr(), Q).reshape(-1, 1)
            Q_old = Q.copy()
            Q = relax * (((1 / k) * self.F + self.S) @ phinew) + (1 - relax) * Q_old
            k_old = k
            k = relax * (k_old * ((Q.sum()) / (Q_old.sum()))) + (1 - relax) * k_old
            err_k = np.absolute(k - k_old)
            err_q = np.absolute(Q - Q_old)
            if err_k <= eps and np.all(err_q <= eps):
                print(i, "Convergence")
                break
            print(i, k, err_k)

        self.k = k
        self.phi = phinew / phinew.sum()

        self.phi = self.expand_flux()
        self.phi = np.nan_to_num(self.phi / self.phi.sum(), nan=0)
        self.phi

        return

    def expand_flux(self) -> np.ndarray:
        """
        Reconstruct the full multigroup flux vector including inactive groups.

        The solver operates on a reduced system where inactive energy groups
        (with zero cross-sections) have been removed. This function rebuilds
        the full flux vector by inserting zeros for those inactive groups.

        Returns
        -------
        phi_full : ndarray
            Full flux vector of size (groups * nb_nodes), where inactive
            groups are filled with zeros.
        """

        if not hasattr(self, "active_groups"):
            raise ValueError("Active groups not defined. Run prep_matrixes() first.")

        N = self.nb_nodes
        G_full = self.groups
        G_eff = self.G_eff

        phi_full = np.zeros((G_full * N))

        for i_new, i_old in enumerate(self.active_groups):
            phi_full[i_old * N : (i_old + 1) * N] = self.phi[i_new * N : (i_new + 1) * N, 0]

        return phi_full
