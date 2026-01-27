rylm
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/chrisiacovella/rylm/workflows/CI/badge.svg)](https://github.com/chrisiacovella/rylm/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/chrisiacovella/rylm/branch/main/graph/badge.svg)](https://codecov.io/gh/chrisiacovella/rylm/branch/main)


Rylm method for identifying local structure

The Rylm method is a computational technique used to identify and classify local structures in condensed matter systems, 
such as liquids and glasses. 
It is based on the calculation of rotationally invariant order parameters derived from spherical harmonics, 
which capture the symmetry properties of the local environment around each particle.

The method was introduced in the paper:

Iacovella, Christopher R. Keys, Aaron S. Horsch, Mark A. Glotzer, Sharon C. (2007). 
“Icosahedral packing of polymer-tethered nanospheres and stabilization of the gyroid phase .” 
PHYSICAL REVIEW E. 75 (040801), 
DOI: 10.1103/PhysRevE.75.040801

The general ideas were further expanded on in: 

Keys, Aaron S. Iacovella, Christopher R. Glotzer, Sharon C. (2011). 
"Characterizing complex particle morphologies through shape matching: Descriptors, applications, and algorithms ." 
JOURNAL OF COMPUTATIONAL PHYSICS. 230 (17).
DOI: 10.1016/j.jcp.2011.04.035

Keys, Aaron S. Iacovella, Christopher R. Glotzer, Sharon C. (2011). 
“Characterizing Structure Through Shape Matching and Applications to Self-Assembly.” 
ANNUAL REVIEW OF CONDENSED MATTER PHYSICS, VOL 2. 2 (), pp 263-285 .
DOI: 10.1146/annurev-conmatphys-062910-140526

Keys, Aaron S. Iacovella, Christopher R. Glotzer, Sharon C. (2010). 
"Harmonic Order Parameters for Characterizing Complex Particle Morphologies"
DOI:10.48550/arXiv.1012.4527


### Installation
To install rylm, you can use pip:

```
pip install rylm
``` 

### Usage
Here is a simple example of how to use rylm to compute local structure order parameters:
```python
import numpy as np
from rylm import Rylm

# Example particle positions
positions = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
                      
# Create Rylm object
rylm = Rylm(include_n_coord=True, include_w=True, frequencies=[4, 6, 8, 10, 12])

fingerprint_test = rylm.calculate(positions)


# calculate the the finger for a few known structures: 
from rylm.data import structures as struct

tetrahedral_fingerprint = rylm.calculate(struct.tetrahedron)
square_planar_fingerprint = rylm.calculate(struct.square_planar)
octahedral_fingerprint = rylm.calculate(struct.octahedron)

# initalize the similarity metric:
similarity_metric = Similarity(metric="euclidean", normalize=True)

rint("\nComparing test points to known structures:")
best_match = {"value": -1, "name": "none"}
for key, fingerprint in library_of_fingerprints.items():
    value = similarity_metric.calculate(fingerprint_test, fingerprint)
    print(f"Similarity between test points and {key} structure: {value}")
    if best_match["value"] == -1 or value < best_match["value"]:
        best_match["value"] = value
        best_match["name"] = key
print("\n")
print(
    f"Best match for test points is {best_match['name']} with similarity {best_match['value']}"
)
```

### Copyright

Copyright (c) 2025, Chris Iacovella


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.11.
