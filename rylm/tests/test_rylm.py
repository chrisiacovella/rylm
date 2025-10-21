"""
Unit and regression test for the rylm package.
"""

# Import package, test suite, and other packages as needed
import sys
import numpy as np

import pytest

import rylm


def test_rylm_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "rylm" in sys.modules


def test_convert_to_spherical_coordinates():
    # This will test the function to convert to spherical coordinates
    # using the icosahedron structure as input
    # This method is used when using scipy to calculate the rylm fingerprint
    ####################################################################################

    from rylm.data.structures import icosahedron
    from rylm.utils import convert_to_spherical_coordinates

    theta, phi, r = convert_to_spherical_coordinates(icosahedron)

    assert theta.shape[0] == icosahedron.shape[0] - 1
    assert phi.shape[0] == icosahedron.shape[0] - 1
    assert r.shape[0] == icosahedron.shape[0] - 1

    assert np.allclose(
        theta,
        np.array(
            [
                1.57079633,
                1.57079633,
                1.57079633,
                1.57079633,
                0.55357436,
                0.55357436,
                2.58801829,
                2.58801829,
                1.01722197,
                1.01722197,
                2.12437069,
                2.12437069,
            ]
        ),
    )
    assert np.allclose(
        phi,
        np.array(
            [
                0.55357436,
                5.72961095,
                3.69516701,
                2.58801829,
                0.0,
                3.14159265,
                3.14159265,
                0.0,
                1.57079633,
                4.71238898,
                4.71238898,
                1.57079633,
            ]
        ),
    )
    # print r formatted such that it has commas between the values
    assert np.allclose(
        r,
        np.array(
            [
                1.90211303,
                1.90211303,
                1.90211303,
                1.90211303,
                1.90211303,
                1.90211303,
                1.90211303,
                1.90211303,
                1.90211303,
                1.90211303,
                1.90211303,
                1.90211303,
            ]
        ),
    )


def test_calculate_Q():
    from rylm.data.structures import icosahedron
    from rylm.utils import convert_to_spherical_coordinates, calculate_Q_scipy

    theta, phi, r = convert_to_spherical_coordinates(icosahedron)

    frequencies = [4, 6]

    # Expected values for Q_l for l=4 and l=6 for a perfect icosahedron
    Q_l_known = {4: 0, 6: 0.663}

    for l in frequencies:
        Q_l = calculate_Q_scipy(theta, phi, l, include_w=False)
        assert np.allclose(
            Q_l, Q_l_known[l], atol=1e-3, rtol=1e-3
        ), f"Q_l for l={l} does not match known value."


def test_rylm_q_fingerpints_icosahedron():
    # this will compare the scipy and freud implementations of the rylm fingerprint
    # This will only calculate the magnitude of the spherical harmonics (Q_l), not Wigner 3j symbols (W_l)

    from rylm.data.structures import icosahedron
    from rylm.rylm import Rylm

    # test that we can modify the frequencies and get the same results
    frequencies = [[4, 6, 8, 10, 12], [4, 6]]

    for freqs in frequencies:
        rylm = Rylm(frequencies=freqs, include_w=False)

        assert rylm._frequencies == freqs, f"Frequencies {freqs} not set correctly."

        fingerprint_scipy = rylm._calculate_fingerprint_scipy(icosahedron)
        fingerprint_freud = rylm._calculate_fingerprint_freud(icosahedron)

        assert (
            fingerprint_scipy.frequencies == fingerprint_freud.frequencies
        ), "Frequencies do not match between scipy and freud implementations."
        assert (
            fingerprint_scipy.include_w == fingerprint_freud.include_w
        ), "include_w does not match between scipy and freud implementations."

        for key in fingerprint_scipy.values.keys():
            assert (
                key in fingerprint_freud.values
            ), f"Key {key} not found in freud fingerprint. It should be"
            assert np.allclose(
                fingerprint_scipy.values[key],
                fingerprint_freud.values[key],
                atol=1e-3,
                rtol=1e-3,
            ), f"Fingerprint for {key} does not match between scipy and freud implementations."


# we will have it loop over including n_coord
@pytest.mark.parametrize("include_n_coord", [True, False])
def test_rylm_q_fingerprint_icosahedron_cutoff(include_n_coord):
    # this will compare the scipy and freud implementations of the rylm fingerprint
    # using a cutoff to ignore points that are far away

    from rylm.data.structures import icosahedron_extended, icosahedron
    from rylm.rylm import Rylm

    # test that we can  use the cutoff.
    # icosahedron_extended is the same as icosahedron, but with a few extra points added
    # we can ignore these extra points by using a cutoff of 2.0

    rylm = Rylm(include_w=False, include_n_coord=include_n_coord)

    fingerprint_scipy = rylm._calculate_fingerprint_scipy(icosahedron)

    fingerprint_scipy_ext = rylm._calculate_fingerprint_scipy(
        icosahedron_extended, cutoff=2.0
    )
    fingerprint_freud_ext = rylm._calculate_fingerprint_freud(
        icosahedron_extended, cutoff=2.0
    )

    assert fingerprint_scipy.frequencies == fingerprint_scipy_ext.frequencies
    assert fingerprint_scipy.frequencies == fingerprint_freud_ext.frequencies
    assert fingerprint_scipy.include_w == fingerprint_scipy_ext.include_w
    assert fingerprint_scipy.include_w == fingerprint_freud_ext.include_w
    assert fingerprint_scipy.include_n_coord == fingerprint_scipy_ext.include_n_coord

    for key in fingerprint_scipy.values.keys():
        assert (
            key in fingerprint_scipy_ext.values
        ), f"Key {key} not found in scipy ext fingerprint. It should be"
        assert (
            key in fingerprint_freud_ext.values
        ), f"Key {key} not found in freud ext fingerprint. It should be"
        assert np.allclose(
            fingerprint_scipy.values[key],
            fingerprint_scipy_ext.values[key],
            atol=1e-3,
            rtol=1e-3,
        ), f"Fingerprint for {key} does not match between scipy and freud implementations."

        assert np.allclose(
            fingerprint_scipy.values[key],
            fingerprint_freud_ext.values[key],
            atol=1e-3,
            rtol=1e-3,
        ), f"Fingerprint for {key} does not match between scipy and freud implementations."


def test_rylm_fingerprint_similarity():
    from rylm.rylm import Fingerprint, Similarity

    # Create two fingerprints with the same frequencies and values
    fingerprint1 = Fingerprint(
        frequencies=[4, 6],
        include_w=True,
        values={
            "q4": np.array([0.1]),
            "q6": np.array([0.2]),
            "w4": np.array([0.3]),
            "w6": np.array([0.4]),
        },
    )
    fingerprint2 = Fingerprint(
        frequencies=[4, 6],
        include_w=True,
        values={
            "q4": np.array([0.1]),
            "q6": np.array([0.2]),
            "w4": np.array([0.3]),
            "w6": np.array([0.4]),
        },
    )
    # Calculate similarity
    eulidian_distance_similarity_metric = Similarity(metric="euclidean", normalize=True)
    similarity = eulidian_distance_similarity_metric.calculate(
        fingerprint1, fingerprint2
    )
    assert similarity == 0.0, "Similarity should be 0 for identical fingerprints."

    # Modify one value in fingerprint2
    fingerprint2.values["q4"] = np.array([0.2])
    # Calculate similarity again
    similarity = eulidian_distance_similarity_metric.calculate(
        fingerprint1, fingerprint2
    )
    assert np.allclose(
        similarity, 0.04761905, atol=1e-5, rtol=1e-5
    ), "Similarity metrics should match."

    # test again where we do not normalize
    eulidian_distance_similarity_metric = Similarity(
        metric="euclidean", normalize=False
    )

    similarity = eulidian_distance_similarity_metric.calculate(
        fingerprint1, fingerprint2
    )
    assert np.allclose(
        similarity, 0.1, atol=1e-5, rtol=1e-5
    ), "Similarity metrics should match."


def test_rylm_similarity():
    from rylm.rylm import Rylm, Similarity
    from rylm.data.structures import icosahedron

    # Create two RylmCluster instances with the same frequencies
    # note we will not include the n_coord in the fingerprint for this test
    rylm = Rylm(frequencies=[4, 6, 8, 10, 12], include_w=True, include_n_coord=False)

    # Calculate fingerprints for a sample structure
    from rylm.data.structures import icosahedron

    fingerprint1 = rylm.calculate(icosahedron)
    # perturb the same icosahedron structure slightly
    np.random.seed(42)  # for reproducibility
    icosahedron2 = icosahedron + np.random.normal(0, 0.01, icosahedron.shape)
    # Calculate fingerprints for the perturbed structure

    fingerprint2 = rylm.calculate(icosahedron2)

    # Calculate similarity
    similarity_metric = Similarity(metric="euclidean", normalize=True)
    similarity = similarity_metric.calculate(fingerprint1, fingerprint2)
    assert np.allclose(
        similarity, 0.010421885
    ), "Similarity should match the expected value."

    icosahedron2 = icosahedron + np.random.normal(0, 0.1, icosahedron.shape)
    # Calculate fingerprints for the perturbed structure

    fingerprint2 = rylm.calculate(icosahedron2)

    # Calculate similarity
    similarity = similarity_metric.calculate(fingerprint1, fingerprint2)
    assert np.allclose(
        similarity, 0.04718723
    ), "Similarity should match the expected value."
