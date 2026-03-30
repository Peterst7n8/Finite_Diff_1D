# Multi-Group Neutron Diffusion Solver

This project implements a finite difference solver for the multi-group neutron diffusion equation, along with a material module for handling nuclear data.

---

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

---

## Installation

```bash
pip install numpy scipy matplotlib periodictable
```

---

## Usage

### Define a material

```python
from Materials import Material

fuel = Material(
    nuclides={'U238': 0.8, 'U235': 0.2, 'O16': 2},
    macro=False,
    density=10.5,
    groups=70
)

fuel.get_xs('U238', 'scat', 'xs_scat.csv')
```

### Build and solve

```python
from solver import Finite_Diff

geom = {fuel: 100.0}
step = [1.0]

solver = Finite_Diff(
    groups=70,
    geom=geom,
    step=step,
    bc_left='reflective',
    bc_right='void'
)

solver.prep_matrixes(plot=False, edge=True)
solver.compute(itext=1000, itint=0, eps=1e-6, relax=0.7)

print(solver.k)
```

---

## Model

Solves the multi-group diffusion equation:

[

* \nabla \cdot D_g \nabla \phi_g + \Sigma_{a,g} \phi_g + \sum_{g'} \Sigma_{s,g \leftarrow g'} \phi_{g'} = \frac{1}{k} \sum_{g'} \nu \Sigma_{f,g' \to g} \phi_{g'}
  ]

---

## Input Data

CSV format:

```
nuclide, group in, group out, mean
```

Supported types:

* `scat`, `abs`, `fiss`, `diff`

---

## Limitations

* Mainly 1D
* No acceleration methods
* Limited validation and error handling

---

## Notes

Educational/research code. Use with caution.
