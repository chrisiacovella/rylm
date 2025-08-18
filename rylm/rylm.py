import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Fingerprint:
    frequencies: List[int]
    include_w: bool = False
    values: Dict[str, np.array] = None


class RylmCluster:
    """

    Class to calculate the Rylm descriptor for a single cluster from a set of points in 3D space.

    Note this is not intented to be used for periodic systems, and will not work correctly with PBC.

    """

    def __init__(
        self,
        frequencies: List[int] = [4, 6, 8, 10, 12],
        include_w: bool = False,
        include_n_coord: bool = True,
    ):
        """
        Initialize the Rylm class with the specified parameters.

        Parameters:
        ----------
        frequencies : List[int], default [4,6,8,10,12]
            A list of integers > 0, representing the frequencies to calculate with spherical harmonics.
            These frequencies should be even integers which are invariant under inversion.
        include_w : bool, default False
            If True, the Wigner3j values will be included in the calculations if using freud.
        include_n_coord : bool, default True
            If True, the coordination number in the fingerprint
        """
        self._frequencies = frequencies
        self._include_w = include_w
        self._include_n_coord = include_n_coord

    def calculate_fingerprint(
        self, points: np.array, cutoff: Optional[float] = None, backend: str = "freud"
    ) -> Dict[str, np.array]:
        """
        Calculate the Rylm descriptor for a set of points in 3D space.

        Parameters:
        ----------
        points : np.array
            An array of shape (n, 3) where n is the number of points, and each point is represented by its (x, y, z) coordinates.
            Note, the first point is considered the origin and will not be included in the descriptor calculation.
        cutoff : Optional[float], default None
            A cutoff distance for the calculation. If provided, it will be used to filter points based on their distance from the origin.
        backend : str, default "freud"
            The backend to use for the calculation. Options are "freud" or "scipy".
            Note the scipy backend will not compute the wigner3j values.

        Returns:
        ----------
        Dict[str, np.array]
            A dictionary where keys are the frequencies (e.g., 'q4', 'q6') and values are the calculated Rylm descriptors for each frequency.
        """
        if backend == "scipy":
            return self._calculate_fingerprint_scipy(points, cutoff)
        elif backend == "freud":
            return self._calculate_fingerprint_freud(points, cutoff)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _calculate_fingerprint_scipy(
        self, points: np.array, cutoff: Optional[float] = None
    ) -> np.array:
        """
        Calculate the Rylm descriptor for a set of points in 3D space.

        This uses scipy's spherical harmonics to compute the descriptor based on the spherical coordinates of the points.
        This will not compute the wigner3j values, as this is not implemented in this method.

        Parameters:
        ----------
        points : np.array
            An array of shape (n, 3) where n is the number of points, and each point is represented by its (x, y, z) coordinates.
            Note, the first point is considered the origin and will not be included in the descriptor calculation.
        cutoff : Optional[float], default None
            A cutoff distance for the calculation. If provided, it will be used to filter points based on their distance from the origin.

        Returns:
        ----------
        Fingerprint
            A dataclass that stores the frequencies, whether wigner3j values are included, and a dictionary of values
            where keys are the frequencies (e.g., 'q4', 'q6) and values are the calculated Rylm descriptors for each frequency.
        """

        # first convert the points to spherical coordinates

        from rylm.utils import convert_to_spherical_coordinates, calculate_Q

        theta, phi, r = convert_to_spherical_coordinates(points)

        if cutoff is not None:
            # filter points based on the cutoff distance
            if not isinstance(cutoff, (int, float)):
                raise TypeError("cutoff must be a number")
            if cutoff < 0:
                raise ValueError("cutoff must be a non-negative number")
            mask = r <= cutoff
            theta = theta[mask]
            phi = phi[mask]

        # calculate the spherical harmonics for each frequency
        fingerprint = Fingerprint(
            frequencies=self._frequencies, include_w=self._include_w
        )

        fingerprint_dict = {}
        for l in self._frequencies:
            if l < 0:
                raise ValueError("Frequency l must be a non-negative integer.")
            ql = calculate_Q(theta, phi, l)
            fingerprint_dict[f"q{l}"] = ql
        if self._include_n_coord:
            fingerprint_dict["n_coord"] = len(theta)
        fingerprint.values = fingerprint_dict
        return fingerprint

    def _calculate_fingerprint_freud(
        self, points: np.array, cutoff: Optional[float] = None
    ) -> np.array:
        """
        Calculate the Rylm descriptor for a set of points in 3D space.

        This uses freud to compute the descriptor based on the spherical coordinates of the points.
        This will also compute the wigner3j values if include_w is True.

        Parameters:
        ----------
        points : np.array
            An array of shape (n, 3) where n is the number of points, and each point is represented by its (x, y, z) coordinates.
            Note, the first point is considered the origin and will not be included in the descriptor calculation.
        cutoff : Optional[float], default None
            A cutoff distance for the calculation. If provided, it will be used to filter points based on their distance from the origin.

        Returns:
        ----------
        Fingerprint
            A dataclass that stores the frequencies, whether wigner3j values are included, and a dictionary of values
            where keys are the frequencies (e.g., 'q4', 'q6) and values are the calculated Rylm descriptors for each frequency.
        """
        import freud

        if not isinstance(points, np.ndarray):
            raise TypeError("points must be a numpy array")

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be a 2D array with shape (n, 3)")

        # this assumes that the points provided are a single cluster without PBC
        # we get the maximum distance between the first point and all other points and use this to
        # create a large enough box to avoid PBC issues, and ensure that all points are in the first neighbor shell

        max_distance = np.max(np.linalg.norm(points[1:] - points[0], axis=1))

        box = freud.box.Box.cube(max_distance * 4)
        system = freud.AABBQuery(box, points)

        # Query the nearest neighbors
        if cutoff is not None:
            if not isinstance(cutoff, (int, float)):
                raise TypeError("cutoff must be a number")
            if cutoff < 0:
                raise ValueError("cutoff must be a non-negative number")
            query_dict = {"r_max": cutoff, "exclude_ii": True}
        else:
            query_dict = {"r_max": max_distance * 1.1, "exclude_ii": True}
            # query_dict = {"num_neighbors": points.shape[0] - 1, "exclude_ii": True}

        nlist = system.query(
            points,
            query_dict,
        ).toNeighborList()

        # We only want the neighbors surrounding the first point (the origin).
        # we will filter the neighbor list to only include those neighbors
        filter_list = [i == 0 for i, j in nlist[:]]
        nlist = nlist.filter(filter_list)

        fingerprint = Fingerprint(
            frequencies=self._frequencies, include_w=self._include_w
        )

        fingerprint_temp = {}
        for l in self._frequencies:
            steinhardt = freud.order.Steinhardt(
                l, wl=self._include_w, wl_normalize=True
            )
            steinhardt.compute(system, neighbors=nlist)

            # get the ql values from the steinhardt object
            # note, that the code sets all points with no neighbors to NaN;
            # since we only want those associated with the first point (the origin), we will just select that
            ql = steinhardt.ql[0]
            if self._include_w:
                wl = steinhardt.particle_order[0]
            fingerprint_temp[f"q{l}"] = ql
            if self._include_w:
                fingerprint_temp[f"w{l}"] = wl
        if self._include_n_coord:
            fingerprint_temp["n_coord"] = (
                len(nlist) - 1
            )  # exclude the first point (the origin)
        fingerprint.values = fingerprint_temp
        return fingerprint


def euclidean_distance(
    fingerprint1: Fingerprint, fingerprint2: Fingerprint, normalize=True
) -> float:
    """
    Calculate the Euclidean distance between two Rylm fingerprints

    Parameters:
    ----------
    fingerprint1 : Fingerprint
        The first Rylm fingerprint.
    fingerprint2 : Fingerprint
        The second Rylm fingerprint.
    normalize : bool, default True
        If True, the similarity will be normalized by the sum of absolute values of the fingerprints.

    Returns:
    ----------
    float
        A similarity score between the two fingerprints.
    """
    # Check if the frequencies match
    # they can be in any order, so we sort them
    if sorted(fingerprint1.frequencies) != sorted(fingerprint2.frequencies):
        raise ValueError("Frequencies of the fingerprints do not match.")

    if fingerprint1.include_w != fingerprint2.include_w:
        raise ValueError("include_w of the fingerprints do not match.")

    # Calculate the similarity as the sum of squared differences
    similarity = 0.0
    normalization = 0.0
    for key in fingerprint1.values.keys():
        if key in fingerprint2.values:
            similarity += (fingerprint1.values[key] - fingerprint2.values[key]) ** 2
            normalization += np.abs(fingerprint1.values[key]) + np.abs(
                fingerprint2.values[key]
            )
    if normalize:
        if normalization == 0:
            raise ValueError("Normalization factor is zero, cannot compute similarity.")
        return np.sqrt(similarity) / normalization

    else:
        return np.sqrt(similarity)
