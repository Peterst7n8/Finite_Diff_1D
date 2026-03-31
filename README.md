# Multi-Group Neutron Diffusion Solver

This project implements a finite difference solver for the multi-group neutron diffusion equation, along with a material module for handling nuclear data.

## Features

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

### Model

Solves the multi-group diffusion equation in 1D heterogenous geometries

### Input Data



### Output Data


## Limitations

* Mainly 1D
* No acceleration 
* Limited validation and error handling

## Notes

Educational/research code. Use with caution.
