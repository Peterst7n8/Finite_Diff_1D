from .Materials import *
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
import matplotlib.pyplot as plt

def extrapolated_distance(mat:Material,h:float,over:bool=True):
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

    distance = np.min(3*mat._diff_coef)

    if h >= 0.5:
        if over:
            return (int(distance/h)+1)*h
        else :
            return int(distance/h)*h
    else :
        if over:
            return (int((0.7104*distance)/h)+1)*h
        else :
            return (int((0.7104*distance)/h))*h


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

    def __init__(self,groups:int = 1,dim:int=1,geom:dict[Material,float] = {Material(nuclides = {'U238':1},macro=False,density=1,groups=1):1},step:np.ndarray = [1],method:str = 'Pierre',bc_left:str='void',bc_right:str='void'):
        """
        groups : int, number of groups in calculation
        geom : dict[Material,size], order of Material and size determines Geometry
        step : list of float, discretization step of each Material
        """

        self.groups = groups
        i=0
        self.materials = np.zeros(len(geom),Material)
        self.region_size = np.zeros(len(geom))
        for mat,val  in geom.items():
            self.materials[i] = mat
            self.region_size[i] = val
            i += 1
        self.step = step

        self.nb_nodes = int(0)
        self.nb_nodes_region = np.zeros(len(geom),int)
        if len(self.materials) != len(self.step):
            if len(self.materials) >= len(self.step):
                raise ValueError("Invalid input : not enough discretization steps or " \
                "too many materials")
            else : 
                raise ValueError("Invalid input : too many discretization steps or too few "
                "Materials")

        if len(self.materials) != 1:
            for i in range(len(self.materials)):
                reste = self.region_size[i] - step[i] * round(self.region_size[i] / step[i])
                tol = 1E-9
                if reste <= tol :
                    self.nb_nodes += int(self.region_size[i]/self.step[i])
                    if i == 0 or i == len(self.materials)-1 :
                        self.nb_nodes_region[i] = self.region_size[i]/self.step[i]
                    else : 
                        self.nb_nodes_region[i] = (self.region_size[i]/self.step[i]) -1
                else :
                    print(self.region_size[i],step[i])
                    print(self.region_size[i] // step[i],self.region_size[i] % step[i])
                    raise ValueError("Discretization step incompatible with domain sizes")
            self.nb_nodes += 1 
        else :
            self.nb_nodes = int(self.region_size[0]/self.step[0])+1
            self.nb_nodes_region = [self.nb_nodes]

        match(dim):
            case 2:
                self.phi = np.ones((int((self.nb_nodes**2)*groups)))
                self.F = sp.lil_matrix((int((self.nb_nodes**2)*groups),int((self.nb_nodes**2)*groups)),dtype=float)
                self.A = sp.lil_matrix((int((self.nb_nodes**2)*groups),int((self.nb_nodes**2)*groups)),dtype=float)
                self.S = sp.lil_matrix((int((self.nb_nodes**2)*groups),int((self.nb_nodes**2)*groups)),dtype=float)
                self.k = 0
                self.method = method

            case default:
                self.phi = np.ones((int(self.nb_nodes*groups)),float)
                self.F = sp.lil_matrix((int(self.nb_nodes*groups),int(self.nb_nodes*groups)),dtype=float)
                self.A = sp.lil_matrix((int(self.nb_nodes*groups),int(self.nb_nodes*groups)),dtype=float)
                self.S = sp.lil_matrix((int(self.nb_nodes*groups),int(self.nb_nodes*groups)),dtype=float)
                self.k = 0
                self.method = method
                self.bc_right = bc_right
                self.bc_left = bc_left


    def prep_matrixes(self,plot,edge) -> None :
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
        for i in range(self.groups):
            self.A[i*self.nb_nodes:(i+1)*self.nb_nodes,i*self.nb_nodes:(i+1)*self.nb_nodes] = self.spatial_matrix_A(i,plot,edge)
        for i in range(self.groups):
            for j in range(self.groups):
                self.S[i*self.nb_nodes:(i+1)*self.nb_nodes,j*self.nb_nodes:(j+1)*self.nb_nodes] = self.spatial_matrix_S(j,i,plot,edge)
                self.F[i*self.nb_nodes:(i+1)*self.nb_nodes,j*self.nb_nodes:(j+1)*self.nb_nodes] = self.spatial_matrix_F(i,j,plot,edge)


        if plot:
            ax = plt.matshow(self.S.toarray())
            plt.colorbar(ax)
            plt.title("Matrice S")
            plt.show()


            ax = plt.matshow(self.F.toarray())
            plt.colorbar(ax)
            plt.title("Matrice F")
            plt.show()

            ax = plt.matshow(self.A.toarray())
            plt.colorbar(ax)
            plt.title("Matrice A")
            plt.show()
                
   
    def A_B_Pi(self,group,plot,edge):
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

        A_mid = np.zeros((int(self.nb_nodes)),dtype=float)
        A_up = np.zeros((int(self.nb_nodes)-1),dtype=float)
        A_down = np.zeros((int(self.nb_nodes)-1),dtype=float)
                
        done = 0
        for i in range(len(self.materials)):
            a = ((2*(self.materials[i]._diff_coef[group])/self.step[i]**2) + self.materials[i]._a_xs[group] + np.sum(self.materials[i]._scat_xs[group,:]))
            D = -(self.materials[i]._diff_coef[group])/(self.step[i]**2)

            if self.nb_nodes_region[i] == 0:
                j=0
                D_left  = self.materials[i]._diff_coef[group]
                D_right = self.materials[i+1]._diff_coef[group]


                # Moyenne harmonique
                D_int = (D_left + D_right)/self.step[i]

                # Mise à jour de la matrice au nœud d’interface
                A_up[done+j] = -D_right / (self.step[i+1]) 
                A_mid[done+j] = D_left/self.step[i] + D_right/self.step[i+1] + 1/2 * (self.step[i]*(self.materials[i]._a_xs[group] + np.sum(self.materials[i]._scat_xs[group,:]))+self.step[i+1]*(self.materials[i+1]._a_xs[group]+ np.sum(self.materials[i+1]._scat_xs[group,:])))
                A_down[done+j-1] = -D_left / (self.step[i])

                done += 1
                continue
                

            for j in range(self.nb_nodes_region[i]):
                if (i == 0 and j==0) : 
                    match self.bc_left:
                        case 'reflective':
                            A_up[done+j] = -(self.materials[i]._diff_coef[group])/(self.step[i])
                            if edge:
                                A_mid[done+j] = (self.materials[i]._diff_coef[group])/(self.step[i]) + self.materials[i]._a_xs[group] + np.sum(self.materials[i]._scat_xs[group,:])
                            else :
                                A_mid[done+j] = (self.materials[i]._diff_coef[group])/(self.step[i])
                        case 'void':
                            A_up[done+j] = D
                        case default:
                            raise ValueError("Unrecognised BC type for left BC")
                elif (i == len(self.materials)-1 and j == self.nb_nodes_region[i]-1):
                    match self.bc_right:
                        case 'reflective':
                            A_down[-1] = -(self.materials[i]._diff_coef[group])/(self.step[i])
                            if edge:
                                A_mid[done+j] = (self.materials[i]._diff_coef[group])/(self.step[i]) + self.materials[i]._a_xs[group] + np.sum(self.materials[i]._scat_xs[group,:])
                            else :
                                A_mid[done+j] = (self.materials[i]._diff_coef[group])/(self.step[i])
                        case 'void':
                            A_down[-1] = D
                        case default:
                            raise ValueError("Unrecognised BC type for right BC")
                else :
                    A_up[done+j] = D
                    A_mid[done+j] = a
                    A_down[done+j-1] = D
                if (i != len(self.materials)-1 and j == self.nb_nodes_region[i]-1):
                    #SI l'on est sur le dernier point d'une région

                    j+=1
                    D_left  = self.materials[i]._diff_coef[group]
                    D_right = self.materials[i+1]._diff_coef[group]


                    # Moyenne harmonique
                    D_int = (D_left + D_right)/self.step[i]

                    # Mise à jour de la matrice au nœud d’interface
                    A_up[done+j] = -D_right / (self.step[i+1]) 
                    A_mid[done+j] = D_left/self.step[i] + D_right/self.step[i+1] + 1/2 * (self.step[i]*(self.materials[i]._a_xs[group] + np.sum(self.materials[i]._scat_xs[group,:]))+self.step[i+1]*(self.materials[i+1]._a_xs[group]+ np.sum(self.materials[i+1]._scat_xs[group,:])))
                    A_down[done+j-1] = -D_left / (self.step[i])
            done += j +1   
             
        A = sp.diags([A_down,A_mid,A_up],[-1,0,1])

        return A

    def spatial_matrix_A(self,group,plot,edge) -> sp.spmatrix :
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
                A = self.A_B_Pi(group,plot,edge)
        return A
 

    def spatial_matrix_S(self,group_in,group_out,plot,edge) -> sp.spmatrix :
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

        S_up = np.zeros((self.nb_nodes-1),float)
        S_mid = np.zeros((self.nb_nodes),float)
        S_down = np.zeros((self.nb_nodes-1),float)
        
        done = 0
        for i in range(len(self.materials)):

            a = self.materials[i]._scat_xs[group_out,group_in]

            if self.nb_nodes_region[i] == 0:
                j=0

                S_up[done+j] = 1/2 * self.step[i+1] * self.materials[i+1]._scat_xs[group_out,group_in]
                S_down[done+j-1] = 1/2 * self.step[i] * self.materials[i]._scat_xs[group_out,group_in]

                done += 1
                continue

            for j in range(self.nb_nodes_region[i]):
                
                
                if ((i == 0 and j == 0) or (i==len(self.materials)-1 and j == self.nb_nodes_region[i]-1)) and not(edge):
                    continue
                elif  (i == 0 and j == 0)  and edge:
                    S_mid[done+j] = 1/2 * self.step[i] * self.materials[i]._scat_xs[group_out,group_in]
                    continue
                elif (i==len(self.materials)-1 and j == self.nb_nodes_region[i]-1) and edge:
                    S_mid[done+j] = 1/2*self.step[i] * self.materials[i]._scat_xs[group_out,group_in]
                    continue

                else :
                    S_mid[done+j] = a 


                if (i != len(self.materials)-1 and j == self.nb_nodes_region[i]-1):
                    #Si l'on arrive à la fin d'une région et que ce n'est pas 
                    #la dernière du domaine 1D


                    j += 1
                    
                    S_mid[done+j] = 0
                    S_up[done+j] = 1/2 * self.step[i+1] * self.materials[i+1]._scat_xs[group_out,group_in]
                    S_down[done+j-1] = 1/2 * self.step[i] * self.materials[i]._scat_xs[group_out,group_in]
                    

            done += j +1  

        S = sp.diags([S_down,S_mid,S_up],[-1,0,1])  

        return S
    
    def spatial_matrix_F(self,group_in,group_out,plot,edge) -> sp.spmatrix :

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

        F_mid = np.zeros((self.nb_nodes),dtype=float)
        F_up = np.zeros((self.nb_nodes-1),dtype=float)
        F_down = np.zeros((self.nb_nodes-1),dtype=float)

        done = 0
        for i in range(len(self.materials)):

            f = self.materials[i]._nu_f_xs[group_in,group_out]

            if self.nb_nodes_region[i] == 0:
                j=0

                F_mid[done+j] = 0
                F_up[done+j] = 1/2 * self.step[i+1] * self.materials[i+1]._nu_f_xs[group_in,group_out]
                F_down[done+j-1] = 1/2 * self.step[i] * self.materials[i]._nu_f_xs[group_in,group_out]

                done+=1
                continue

            for j in range(self.nb_nodes_region[i]):

                if ((i == 0 and j == 0) or (i==len(self.materials)-1 and j == self.nb_nodes_region[i]-1)) and not(edge):
                    continue
                elif  (i == 0 and j == 0)  and edge:
                    F_mid[done+j] = 1/2 * self.materials[i]._nu_f_xs[group_in,group_out]
                elif (i==len(self.materials)-1 and j == self.nb_nodes_region[i]-1) and edge:
                    F_mid[done+j] = 1/2 * self.materials[i]._nu_f_xs[group_in,group_out]
                elif self.method == 'A-H':
                    F_up[done+j] = 1/2 * self.materials[i]._nu_f_xs[group_in,group_out]
                    F_down[done+j-1] = 1/2 * self.materials[i]._nu_f_xs[group_in,group_out]
                else : 
                    F_mid[done+j] = f

                if(i != len(self.materials)-1 and j == self.nb_nodes_region[i]-1):

                    j+=1

                    F_mid[done+j] = 0
                    F_up[done+j] = 1/2 * self.step[i+1] * self.materials[i+1]._nu_f_xs[group_in,group_out]
                    F_down[done+j-1] = 1/2 * self.step[i] * self.materials[i]._nu_f_xs[group_in,group_out]
                    
            done += j +1 
        
        F = sp.diags([F_down,F_mid,F_up],[-1,0,1])

        return F
    
    def compute(self,itext,itint,eps,relax:float):
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
        k = 1
        Q =  self.phi @ (self.S + self.F)

        phinew = np.zeros((self.groups*self.nb_nodes),float)
        err_k = float(0)
        err_q = float(0)
        for i in range(itext):
            phinew = spla.spsolve(self.A.tocsr(),Q)
            Q_old = Q.copy()
            Q  = relax*(phinew @ ((1/k)*self.F + self.S)) + (1-relax)*Q_old
            k_old = k 
            k = relax*(k_old * ((Q.sum())/(Q_old.sum()))) + (1-relax)*k_old
            err_k = np.absolute(k - k_old)
            err_q = np.absolute(Q - Q_old)
            if (err_k <= eps and np.all(err_q <= eps)):
                print(i,'Convergence')
                break
            print(i,k,err_k)

        self.k = k
        self.phi = phinew/phinew.sum()



        return 
    



        
