# Multi-Group Neutron Diffusion Solver

1. [Presentation](#I.Presentation)
2. [Prerequisite](#II.Prerequisite)
3. [Installation](#III.Installation)
4. [Structure](#IV.Structure)
5. [Features](#V.Features)
6. [Limitations](#VI.Limitations)
7. [Notes](#VII.Notes)


## I.Presentation

This project implements a finite difference solver for the multi-group neutron diffusion equation, along with a material module for handling nuclear data.


## II.Prerequisite

- **Python** : >= 3.11
- **uv** : environment and dependecies gestionary
  https://docs.astral.sh/uv/


## III.Installation

```bash
git clone https://github.com/Peterst7n8/Finite_Diff_1D.git
cd Finite_Diff_1D
uv sync
$ uv run python3 examples/fuel_slab_water_reflector.py
```

## IV.Structure

```console
.
├── examples/
├── src/
├── tests/
├── .gitignore
├── pyproject.toml
└── README.md
3 directories, 4 files
```

## V.Features

### Solver

* Multi-group neutron diffusion (1D)
* Multi-region geometries
* Sparse matrix formulation (`scipy.sparse`)
* Power iteration eigenvalue solver (k-effective)
* Boundary conditions: `void`, `reflective`
* Interface treatment between materials

### Materials

* Isotopic composition definition
* Automatic normalization of atomic fractions
* Molar mass computation
* Microscopic → macroscopic cross section conversion
* CSV-based nuclear data loading

### Physics solved

Multi-group neutron diffusion equation in 1D heterogenous geometries


## VI.Limitations

* Mainly 1D
* No acceleration 
* Limited validation and error handling


## VII.Notes

Educational/research code. Use with caution.
