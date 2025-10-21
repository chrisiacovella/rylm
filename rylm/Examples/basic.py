# Basic example of how to use the rylm package to identify the structure of a set of points.

import numpy as np
from rylm.rylm import Rylm, Similarity

# define a set of test points in a square planar arrangement
# note the first point is considered the origin of the set of points
test_points = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)

# Initialize the Rylm class that is used to calculate fingerprint
# By default the frequencies used are [4, 6, 8, 10, 12], and that we will include the wigner3j values
# Also by default, the coordination number (n_coord) is not included in the fingerprint as well; this can be useful in
# cases where we want to exclude similar structures with different coordination numbers.

rylm = Rylm(include_n_coord=True, include_w=True, frequencies=[4, 6, 8, 10, 12])

# Calculate the fingerprint for the test points
fingerprint_test = rylm.calculate(test_points)

# we can print the fingerprint out to see what it contains
# It is a dataclass that contains some of the useful metadata about the fingerprint,
# so we can check that when comparing fingerprints later that they are computed on an equivalent basis
# The values of the fingerprint itself are stored in a dictionary called 'values' within the fingerprint object
print("Fingerprint for the test points:")
print(fingerprint_test)

# output should be similar to the following:
"""
frequencies=[4, 6, 8, 10, 12], 
include_w=True, 
include_n_coord=True, 
values={
    'q4': np.float32(0.82915616), 
    'w4': np.float32(0.124970965), 
    'q6': np.float32(0.5863019), 
    'w6': np.float32(-0.0072146733), 
    'q8': np.float32(0.7979466), 
    'w8': np.float32(0.06380072), 
    'q10': np.float32(0.61396503), 
    'w10': np.float32(-0.016309233), 
    'q12': np.float32(0.78281087), 
    'w12': np.float32(0.042356532), 
    'n_coord': 4
}
"""

# Now we can compare the fingerprint of the test points to known fingerprints of common structures
# For this example, we will compare to fingerprints of square planar, tetrahedral, and octahedral structures
from rylm.data import structures as struct

tetrahedral_fingerprint = rylm.calculate(struct.tetrahedron)
square_planar_fingerprint = rylm.calculate(struct.square_planar)
octahedral_fingerprint = rylm.calculate(struct.octahedron)
library_of_fingerprints = {
    "tetrahedral": tetrahedral_fingerprint,
    "square planar": square_planar_fingerprint,
    "octahedral": octahedral_fingerprint,
}

# The original Rylm paper used a Euclidean distance metric to compare fingerprints
# Where a value of 0 is a perfect match.
# Here we will use the Similarity class to calculate the similarity between fingerprints
# Note, by default the Similarity class normalizes the similarity value between 0 and 1
similarity_metric = Similarity(metric="euclidean", normalize=True)

print("\nComparing test points to known structures:")
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


# Expected output should indicate that the best match is the square planar structure with similarity of 0,
# given that the square planar points are identical to the test points provided.

np.random.seed(42)  # for reproducibility

# To better test, we can perturb the points slightly and see if the best match remains the same
perturbed_test_points = test_points + np.random.normal(0, 0.05, test_points.shape)
fingerprint_perturbed = rylm.calculate(perturbed_test_points)

best_match_perturbed = {"value": -1, "name": "none"}
print("\nComparing perturbed test points to known structures:")
for key, fingerprint in library_of_fingerprints.items():
    value = similarity_metric.calculate(fingerprint_perturbed, fingerprint)
    print(f"Similarity between perturbed test points and {key} structure: {value}")
    if best_match_perturbed["value"] == -1 or value < best_match_perturbed["value"]:
        best_match_perturbed["value"] = value
        best_match_perturbed["name"] = key
print("\n")
print(
    f"Best match for perturbed test points is {best_match_perturbed['name']} with similarity {best_match_perturbed['value']}"
)

# Here the best match should still be the square planar structure, though the similarity value will be slightly higher
# than 0 due to the perturbation of the points.


# let us consider a system from the tmqm dataset which I visually identified as square planar
# should be square planar, identifier ABAFOZ
test_points = np.array(
    [
        [0.30918241, 0.42624341, 0.37986646],
        [0.1073847, 0.40108952, 0.45357848],
        [0.23021752, 0.61345033, 0.30564696],
        [0.50053981, 0.49386629, 0.33569512],
        [0.34640283, 0.22676314, 0.42446571],
    ]
)

fingerprint_test1 = rylm.calculate(test_points)

# The follow visually appears octahedral, identifier CECBES
test_points2 = np.array(
    [
        [1.02725949, 0.13879648, 0.38548378],
        [1.14499885, 0.23049694, 0.54779695],
        [1.22150795, 0.08531673, 0.30058462],
        [1.0110957, 0.32119666, 0.30138697],
        [0.9162874, 0.07910506, 0.22024935],
        [1.01663292, -0.04125328, 0.47262867],
        [0.83552833, 0.17913083, 0.47278508],
    ]
)

fingerprint_test2 = rylm.calculate(test_points2)

fingerprints = {"test_system1": fingerprint_test1, "test_system2": fingerprint_test2}

similarity = Similarity(metric="euclidean", normalize=True)

for name, fingerprint in fingerprints.items():
    best_match = {"value": -1, "name": "none"}
    for key, ref_fingerprint in library_of_fingerprints.items():
        value = similarity.calculate(fingerprint, ref_fingerprint)
        print(f"Similarity between {name} and {key} structure: {value}")
        if best_match["value"] == -1 or value < best_match["value"]:
            best_match["value"] = value
            best_match["name"] = key
    print(
        f"\nBest match for {name} is {best_match['name']} with similarity {best_match['value']}\n"
    )
