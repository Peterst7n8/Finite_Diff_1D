import json as js
import numpy as np
import periodictable as pt
import scipy.constants as scpcst
import uuid

"""
Author : Pierre Boussemart

Material Module for multi-group neutron diffusion calculations.

This module defines a 'Material' class used to represent a material in
neutron diffusion simulations. It handles both microscopic and macroscopic 
cross sections and provides utilities to compute macroscopic quantities.

Physical context:
- Multi-group neutron diffusion
- Cross section types : scattering, fission, absorption
- Diffusion coefficients
- Supports conversion from microscopic to macroscopic data

Main features :
- Definition of materials via isotopic composition
- Automatic normalization of atomic fractions 
- Molar mass computation from isotopic data
- Conversion from microscopic XS to macroscopic data
- Loading cross sections and diffusion coefficients from CSV files 

Conventions :
- Densities in g/cm³
- Concentrations in cm⁻³
- Cross sections in barns (microscopic) or cm⁻¹ (macroscopic)
- Energy groups indexed from 0 to G-1

Dependencies :
- numpy
- scipy
- periodictable
- json
- uuid

Notes :
This module is intended for educational and research purposes only and can contain errors.
"""


class Material:
    """
    Represents a material used in neutron transport or diffusion calculations.

    This class stores:
    - Nuclide composition (atomic fractions)
    - Density
    - Number of energy groups
    - Microscopic or macroscopic cross sections

    It provides methods to compute concentrations and to load or build
    cross sections (scattering, absorption, fission, diffusion).
    """

    def __init__(self, nuclides: dict[str, float] = None, macro: bool = False, density: float = 0, groups: int = 1):
        """
        Initialize a Material instance.

        Parameters
        ----------
        nuclides : dict[str, float], optional
            Dictionary of nuclides with atomic fractions.
        macro : bool, optional
            If True, cross sections are treated as macroscopic.
        density : float, optional
            Material density in g/cm^3.
        groups : int, optional
            Number of energy groups.
        """

        self._macro = macro
        self._groups = groups
        self._nuclides = nuclides
        self._density = density
        if not self._density or self._density <= 0:
            self._density = None
        else:
            self._density = density

        if self._nuclides != None:
            self._normalize_nuclides()

        self._scat_xs = np.zeros((groups, groups), float)
        self._a_xs = np.zeros((self._groups), float)
        self._nu_f_xs = np.zeros((groups, groups), float)
        self._diff_coef = np.zeros((groups), float)

        self._hash_id = int(uuid.uuid4())

    # ---------------------------------------------------------------------------------------------------------------------------------------------- #
    # property                                                                                                                                       #
    # ---------------------------------------------------------------------------------------------------------------------------------------------- #
    @property
    def scat_xs(self):
        return self._scat_xs

    @property
    def a_xs(self):
        return self._a_xs

    @property
    def nu_f_xs(self):
        return self._nu_f_xs

    @property
    def diff_coef(self):
        return self._diff_coef

    # ---------------------------------------------------------------------------------------------------------------------------------------------- #
    # private metthods                                                                                                                               #
    # ---------------------------------------------------------------------------------------------------------------------------------------------- #
    def _normalize_nuclides(self):
        """
        Normalize atomic fractions so their sum equals 1.

        Raises
        ------
        ValueError
            If the total fraction is non-positive.
        """
        total = sum(self._nuclides.values())
        if total <= 0:
            raise ValueError("Composition problem")
        return {isotope: fraction / total for isotope, fraction in self._nuclides.items()}

    def _get_conc(self, nuclide: str = "", units: str = ""):
        """
        Compute the atomic concentration of a given nuclide.

        Parameters
        ----------
        nuclide : str
            Target nuclide.
        units : str
            Output units: 'cm-3' (default) or 'b/cm'.

        Returns
        -------
        float
            Concentration in the requested units.

        Raises
        ------
        ValueError
            If density or composition is not defined, or nuclide is unknown.
        """
        if self._density == None:
            raise ValueError("density isn't set")

        if self._nuclides == None:
            raise ValueError("Nuclides and fractions list doesn't exist")

        if not (nuclide in self._nuclides.keys()):
            raise ValueError(f"Unrecognised nuclide: {nuclide}")

        conc = self._nuclides[nuclide] * ((self._density * scpcst.Avogadro) / self._get_molar_mass())

        if units == "b/cm":
            conc = conc / 1e24

        return conc

    def _get_molar_mass(self):
        """
        Compute the molar mass of the material.

        Returns
        -------
        float
            Molar mass in g/mol.
        """
        total_fraction = sum(self._nuclides.values())
        if total_fraction <= 0:
            raise ValueError("Input composition and atomic fractions prevent computation of the molar mass")

        molar_mass = 0.0
        for isotope, fraction in self._nuclides.items():
            molar_mass += fraction * self._get_isotope_mass(isotope)

        return molar_mass

    @staticmethod
    def _column_index(lines: list[str], key: str) -> int:
        """
        Return the column index corresponding to a given key in a CSV-like list of lines.

        Parameters
        ----------
        lines : list[str]
            Lines of a CSV file (including header).
        key : str
            Column name to search for.

        Returns
        -------
        int
            Index of the column where the key is found.
        """
        for line in lines:
            for j, field in enumerate(line.split(",")):
                if key in field:
                    return j

    @staticmethod
    def _get_isotope_mass(isotope_key: str):
        """
        Retrieve the molar mass of a given isotope from its name.

        Parameters
        ----------
        isotope_key : str
            Isotope name (e.g., 'U238', 'O16').

        Returns
        -------
        float
            Molar mass of the isotope in g/mol.

        Raises
        ------
        ValueError
            If the isotope mass is unknown.
        """
        symbol = "".join(filter(str.isalpha, isotope_key))
        mass_number = int("".join(filter(str.isdigit, isotope_key)))

        element = getattr(pt, symbol)
        if mass_number != 0:
            isotope = element[mass_number]
        else:
            if element.mass is None:
                raise ValueError(f"Unknown mass for {isotope_key}")
            else:
                return element.mass

        if isotope.mass is None:
            raise ValueError(f"Unknown mass for {isotope_key}")

        return isotope.mass

    # ---------------------------------------------------------------------------------------------------------------------------------------------- #
    # public metthods                                                                                                                                #
    # ---------------------------------------------------------------------------------------------------------------------------------------------- #
    def clone(self):
        """
        Create a deep copy of the material.

        Returns
        -------
        Material
            A new Material instance with identical properties.
        """
        new_mat = Material(nuclides=self._nuclides.copy() if self._nuclides else None, macro=self._macro, density=self._density, groups=self._groups)

        new_mat._scat_xs = self._scat_xs.copy()
        new_mat._a_xs = self._a_xs.copy()
        new_mat._nu_f_xs = self._nu_f_xs.copy()
        new_mat._diff_coef = self._diff_coef.copy()

        return new_mat

    def get_macro_xs(self, type: str = "scat", filename: str = "", xs_type: str = "macro"):
        """
        Compute macroscopic cross sections from a CSV file.

        Parameters
        ----------
        type : str
            Type of cross section: 'scat', 'abs', 'fiss', or 'diff'.
        filename : str
            Path to the CSV file.
        xs_type : str
            'macro' for direct values, 'micro' for conversion using concentrations.
        """
        file_xs = open(filename, "r")
        lines = file_xs.readlines()

        if type == "scat":
            mean = self._column_index(lines, "mean")
            group_in = self._column_index(lines, "group in")
            group_out = self._column_index(lines, "group out")
            nuclide_ind = self._column_index(lines[:1], "nuclide")

            for n, line in enumerate(lines[1:]):
                g_in = int(line.split(",")[group_in]) - 1
                g_out = int(line.split(",")[group_out]) - 1

                if xs_type == "macro":
                    self._scat_xs[g_in, g_out] += float(line.split(",")[mean])
                elif xs_type == "micro":
                    nuclide = line.split(",")[nuclide_ind]

                    print("oayo", line)

                    conc = self._get_conc(nuclide, "b/cm")
                    self._scat_xs[g_in, g_out] += conc * float(line.split(",")[mean])

            for n in range(self._scat_xs.shape[0]):
                self._scat_xs[n, n] = 0

        elif type == "fiss":
            mean = self._column_index(lines, "mean")
            group_in = self._column_index(lines, "group in")
            group_out = self._column_index(lines, "group out")
            nuclide_ind = self._column_index(lines[:1], "nuclide")
            for n, line in enumerate(lines[1:]):
                g_in = int(line.split(",")[group_in]) - 1
                g_out = int(line.split(",")[group_out]) - 1
                if xs_type == "macro":
                    self._nu_f_xs[g_in, g_out] += float(line.split(",")[mean])
                elif xs_type == "micro":
                    nuclide = line.split(",")[nuclide_ind]
                    conc = self._get_conc(nuclide, "b/cm")
                    self._nu_f_xs[g_in, g_out] += conc * float(lines.split(",")[mean])

        elif type == "abs":
            mean = self._column_index(lines, "mean")
            nuclide_ind = self._column_index(lines[:1], "nuclide")
            grp = 0
            for n, line in enumerate(lines[1:]):
                if xs_type == "macro":
                    self._a_xs[grp] += float(line.split(",")[mean])
                elif xs_type == "micro":
                    nuclide = line.split(",")[nuclide_ind]
                    conc = self._get_conc(nuclide, "b/cm")
                    self._a_xs[grp] += conc * float(line.split(",")[mean])
                grp += 1

        else:
            raise ValueError(f"Unknown cross section type: {type} must be 'scat', 'abs', 'fiss', or 'diff'")

    def get_diff(self, filename: str = ""):
        """
        Load diffusion coefficients from a CSV file.

        Parameters
        ----------
        filename : str
            Path to the CSV file.
        """
        file_xs = open(filename, "r")
        lines = file_xs.readlines()
        mean = self._column_index(lines, "mean")
        for n, line in enumerate(lines[1:]):
            self._diff_coef[n] = float(line.split(",")[mean])

    def get_xs(self, nuclide: str = "", type: str = "scat", filename: str = "") -> None:
        """
        Load microscopic cross sections from a CSV file for a specific nuclide and update the material.

        Parameters
        ----------
        nuclide : str
            Target nuclide.
        type : str
            Type of cross section: 'scat', 'abs', 'diff', or 'fiss'.
        filename : str
            Path to the CSV file.
        """
        file_xs = open(filename, "r")
        conc = self._get_conc(nuclide, "b/cm")
        lines = file_xs.readlines()
        if type == "scat":
            mean = self._column_index(lines, "mean")
            group_in = self._column_index(lines, "group in")
            group_out = self._column_index(lines, "group out")
            nuclide_ind = self._column_index(lines, "nuclide")
            nuclide_scat = np.zeros((self._groups, self._groups), float)
            for n, line in enumerate(lines[1:]):
                if line.split(",")[nuclide_ind] == nuclide:
                    g_in = int(line.split(",")[group_in]) - 1
                    g_out = int(line.split(",")[group_out]) - 1
                    nuclide_scat[g_in, g_out] = line.split(",")[mean]
            for n in range(np.shape(nuclide_scat)[0]):
                nuclide_scat[n, n] = 0
            self._scat_xs += conc * nuclide_scat

        elif type == "fiss":
            mean = self._column_index(lines, "mean")
            group_in = self._column_index(lines, "group in")
            group_out = self._column_index(lines, "group out")
            nuclide_ind = self._column_index(lines, "nuclide")
            nuclide_fiss = np.zeros((self._groups, self._groups), float)
            for n, line in enumerate(lines[1:]):
                if line.split(",")[nuclide_ind] == nuclide:
                    g_in = int(line.split(",")[group_in]) - 1
                    g_out = int(line.split(",")[group_out]) - 1
                    nuclide_fiss[g_in, g_out] = line.split(",")[mean]
            self._nu_f_xs += conc * nuclide_fiss

        elif type == "abs":
            mean = self._column_index(lines, "mean")
            nuclide_ind = self._column_index(lines, "nuclide")
            grp = 0
            nuclide_abs = np.zeros(self._groups, float)
            for n, line in enumerate(lines[1:]):
                if line.split(",")[nuclide_ind] == nuclide:
                    nuclide_abs[grp] = line.split(",")[mean]
                    grp += 1
            self._a_xs += conc * nuclide_abs

        else:
            raise ValueError(f"Unknown cross section type: {type} must be 'scat', 'abs', 'fiss', or 'diff'")

    # ---------------------------------------------------------------------------------------------------------------------------------------------- #
    # interface                                                                                                                                      #
    # ---------------------------------------------------------------------------------------------------------------------------------------------- #
    def __str__(self):
        """
        Return a string representation of the material.
        """
        tab = str("\n")
        for k, v in self._nuclides.items():
            tab += f"{k} = {v}"
            tab += "\n"

        return f"nuclides : {tab}\nmacro : {self._macro}\ndensity : {self._density}\ngroups = {self._groups}\n"

    def __eq__(self, other):
        """
        Compare two Material objects for equality.

        Returns
        -------
        bool
        """
        if not (isinstance(other, Material)):
            return False

        return (
            self._nuclides == other._nuclides
            and self._macro == other._macro
            and self._density == other._density
            and self._groups == other._groups
            and self._hash_id == other._hash_id
        )

    def __hash__(self):
        """
        Return a hash value for the material.

        Returns
        -------
        int
        """
        return hash((js.dumps(self._nuclides, sort_keys=True), self._groups, self._density))


if __name__ == "__main__":
    # TODO: faire des tests unitaires et d'intégration à la place de ce main

    s = Material(nuclides={"U238": 0.8, "U235": 0.2, "O16": 2}, macro=False, density=10.5, groups=70)
    # s.get_xs("U238", "scat", "./solv_num/mgxs_nuclide/xs_scat_core.csv")
    # s.get_xs("U235", "scat", "./solv_num/mgxs_nuclide/xs_scat_core.csv")
    # s.get_xs("O16", "scat", "./solv_num/mgxs_nuclide/xs_scat_core.csv")
    # print(s._scat_xs)

    test = Material(nuclides={"U238": 0.8, "U235": 0.2, "O16": 2}, macro=True, groups=70, density=10.5)
    # test.get_macro_xs("scat", "./solv_num/mgxs_nuclide/xs_scat_core.csv", "micro")
    # print(test._scat_xs)
