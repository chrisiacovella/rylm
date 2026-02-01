import numpy as np
from typing import Optional
from dataclasses import dataclass
from collections import OrderedDict

@dataclass
class Fingerprint:
    """
    Dataclass to store the Rylm fingerprint values and other metadata

    Parameters:
    ----------
    frequencies : list[int]
        A list of integers > 0, representing the frequencies used in the calculation.
    include_w : bool
        If True, the Wigner3j values are included in the calculations and fingerprints.
    include_n_coord : bool
        If True, the coordination number is included in the fingerprint.
    values : dict[str, np.array]
        A dictionary where keys are the frequencies (e.g., 'q4', 'q6', 'w4', 'w6') and values are the calculated Rylm descriptors for each frequency.

    """

    frequencies: list[int]
    include_w: bool = True
    include_n_coord: bool = True
    values: OrderedDict[str, np.array] = None

    def to_numpy(self) -> np.array:
        """
        Convert the fingerprint values to a single numpy array for similarity calculations.

        Returns:
        ----------
        np.ndarray
            A 1D numpy array containing all the fingerprint values concatenated.
        """
        array_list = []
        for key in self.values:
            array_list.append(np.array(self.values[key]).flatten())
        return np.concatenate(array_list)


class Rylm:
    """

    Class to calculate the Rylm descriptor for a single cluster of points in 3D space.

    Note this is not intended to be used for periodic systems and will not work correctly with PBC.

    """

    def __init__(
        self,
        frequencies: list[int] = [2, 4, 6, 8, 10, 12],
        include_w: bool = True,
        include_n_coord: bool = True,
    ):
        """
        Initialize the Rylm class with the specified parameters.

        Parameters:
        ----------
        frequencies : list[int], default [2, 4,6,8,10,12]
            A list of integers > 0, representing the frequencies to calculate with spherical harmonics.
            These frequencies should be even integers which are invariant under inversion.
            Regardless of the order provided, they will be sorted internally from smallest to largest.
        include_w : bool, default True
            If True, the Wigner3j values will be included in the calculations if using freud.
        include_n_coord : bool, default True
            If True, the coordination number will be included in the fingerprint
        """
        for l in frequencies:
            if l < 0:
                raise ValueError("Frequency l must be a non-negative integer.")
            if l % 2 != 0:
                raise ValueError("Frequency l must be an even integer.")

        self._frequencies = frequencies

        # sort the frequencies to ensure they are from smallest to largest for consistency
        self._frequencies.sort()

        self._include_w = include_w
        self._include_n_coord = include_n_coord

    def calculate(
        self, points: np.array, cutoff: Optional[float] = None, backend: str = "freud", center_index: int = 0,
    ) -> Fingerprint:
        """
        Calculate the Rylm descriptor for a set of points in 3D space.

        Parameters:
        ----------
        points : np.array
            An array of shape (n, 3) where n is the number of points, and each point is represented by its (x, y, z) coordinates.
            Note, the origin of the complex is set with ``center_index`` and will not be included in the descriptor calculation.
        cutoff : Optional[float], default None
            A cutoff distance for the calculation. If provided, it will be used to filter points based on their distance from the origin (i.e., first point).
        backend : str, default "freud"
            The backend to use for the calculation. Options are "freud" or "scipy".
            Note the scipy backend will not compute the wigner3j values.
        center_index : int, default=0
            The index of the central atom.

        Returns:
        ----------
        Fingerprint
            A dataclass that stores the frequencies, whether wigner3j values are included, and a dictionary of values
            where keys are the frequencies (e.g., 'q4', 'q6) and values are the calculated Rylm descriptors for each frequency.
        """
        
        if not isinstance(points, np.ndarray):
            raise TypeError("points must be a numpy array")

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be a 2D array with shape (n, 3)")

        if center_index < 0 or center_index >= len(points):
            raise ValueError(f"center_index {center_index} is out of bounds for the points array")

        # Reorder points so that the center_index is the first row
        points = np.vstack([points[center_index], np.delete(points, center_index, axis=0)])

        if backend == "scipy":
            return self._calculate_fingerprint_scipy(points, cutoff)
        elif backend == "freud":
            return self._calculate_fingerprint_freud(points, cutoff)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _calculate_fingerprint_scipy(
        self, points: np.array, cutoff: Optional[float] = None
    ) -> Fingerprint:
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

        from rylm.utils import convert_to_spherical_coordinates, calculate_Q_scipy

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
            frequencies=self._frequencies,
            include_w=self._include_w,
            include_n_coord=self._include_n_coord,
        )

        fingerprint_dict = OrderedDict()
        for l in self._frequencies:
            ql = calculate_Q_scipy(theta, phi, l)
            fingerprint_dict[f"q{l}"] = ql
        if self._include_n_coord:
            fingerprint_dict["n_coord"] = len(theta)
        fingerprint.values = fingerprint_dict
        return fingerprint

    def _calculate_fingerprint_freud(
        self, points: np.array, cutoff: Optional[float] = None
    ) -> Fingerprint:
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

        box = freud.box.Box.cube(max_distance * 5)
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
            frequencies=self._frequencies,
            include_w=self._include_w,
            include_n_coord=self._include_n_coord,
        )

        fingerprint_temp = OrderedDict()
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
            fingerprint_temp["n_coord"] = len(
                nlist
            )  # exclude the first point (the origin)
        fingerprint.values = fingerprint_temp
        return fingerprint


class Similarity:
    """
    Class to calculate similarity between two Rylm fingerprints.

    """

    def __init__(self, metric: str = "euclidean", normalize: bool = True):
        """
        Initialize the Similarity class with the specified metric.

        Parameters:
        ----------
        metric : str, default "euclidean"
            The similarity metric to use. Currently, "euclidean", "manhattan", and "dot_product" are supported.
        normalize : bool, default True
            If True, the similarity will be normalized by the sum of absolute values of the fingerprints.

        """
        if metric == "euclidean":
            self.similarity_function = self._euclidean_similarity
        elif metric == "manhattan":
            self.similarity_function = self._manhattan_similarity
        elif metric == "dot_product":
            self.similarity_function = self._dot_product_similarity
        else:
            raise ValueError(f"Unknown metric: {metric}")

        self._metric = metric
        self._normalize = normalize

    def calculate(self, fingerprint1: Fingerprint, fingerprint2: Fingerprint) -> float:
        """
        Calculate the similarity between two Rylm fingerprints.

        Parameters:
        ----------
        fingerprint1 : Fingerprint
            The first Rylm fingerprint.
        fingerprint2 : Fingerprint
            The second Rylm fingerprint.
        Returns:
        ----------
        float
            A similarity score between the two fingerprints.

        """
        # Check if the frequencies match
        # they can be in any order, so we sort them
        if sorted(fingerprint1.frequencies) != sorted(fingerprint2.frequencies):
            raise ValueError("Frequencies of the fingerprints do not match.")

        # Check if include_w and include_n_coord match
        if fingerprint1.include_w != fingerprint2.include_w:
            raise ValueError("include_w of the fingerprints do not match.")

        if fingerprint1.include_n_coord != fingerprint2.include_n_coord:
            raise ValueError("include_n_coord of the fingerprints do not match.")

        # Check if the values keys match
        if fingerprint1.values.keys() != fingerprint2.values.keys():
            raise ValueError("Values keys of the fingerprints do not match.")

        return self.similarity_function(
            fingerprint1, fingerprint2, normalize=self._normalize
        )

            
    def _euclidean_similarity(
        self, fingerprint1: Fingerprint, fingerprint2: Fingerprint, normalize=True
    ) -> float:
        """
        Calculate the Euclidean similarity between two Rylm fingerprints.
        
        The similarity is (1-distance), where distance he Euclidean distance.
        This ensures similarity values are in the range (0, 1], with 1 indicating identical fingerprints.

        Parameters:
        ----------
        fingerprint1 : Fingerprint
            The first Rylm fingerprint.
        fingerprint2 : Fingerprint
            The second Rylm fingerprint.
        normalize : bool, default True
            If True, the distance will be normalized by the sum of absolute values of the fingerprints.

        Returns:
        ----------
        float
            A similarity score between the two fingerprints, where higher values indicate greater similarity.
        """

        fp1_value = fingerprint1.to_numpy()
        fp2_value = fingerprint2.to_numpy()

        distance = np.sqrt(np.sum((fp1_value - fp2_value) ** 2, axis=0))
        if normalize:
            normalization = np.sum(np.abs(fp1_value) + np.abs(fp2_value))
            distance = distance / normalization

        similarity = 1.0 - distance

        return similarity


    def _manhattan_similarity(
        self, fingerprint1: Fingerprint, fingerprint2: Fingerprint, normalize=True
    ) -> float:
        """
        Calculate the Manhattan similarity between two Rylm fingerprints.

        To scale values between 0 and 1, where 1 is an ideal match,
        the similarity is defined as 1/(1 + manhattan distance).

        Normalization, if selected, is done by dividing the distance by the sum of absolute values of the fingerprints.

        Parameters:
        ----------
        fingerprint1 : Fingerprint
            The first Rylm fingerprint.
        fingerprint2 : Fingerprint
            The second Rylm fingerprint.
        normalize : bool, default True
            If True, the distance will be normalized by the sum of absolute values of the fingerprints.

        Returns:
        ----------
        float
            A similarity score between the two fingerprints, where higher values indicate greater similarity.
        """

        vec1 = fingerprint1.to_numpy()
        vec2 = fingerprint2.to_numpy()

        d_T = np.sum(np.abs(vec1 - vec2))
        if normalize:
            normalization = np.sum(np.abs(vec1) + np.abs(vec2))
            d_T = d_T / normalization

        return 1/(1+d_T)


    def _dot_product_similarity(
        self, fingerprint1: Fingerprint, fingerprint2: Fingerprint, normalize=True
    ) -> float:
        """
        Calculate the dot product similarity between two Rylm fingerprints.
        
        Values are shifted, such that range is [0, 1], where 1 indicates identical orientation.

        Parameters:
        ----------
        fingerprint1 : Fingerprint
            The first Rylm fingerprint.
        fingerprint2 : Fingerprint
            The second Rylm fingerprint.

        Returns:
        ----------
        float
            A similarity score between the two fingerprints, where higher values indicate greater similarity.
        """

        vec1 = fingerprint1.to_numpy()
        vec2 = fingerprint2.to_numpy()

        dot_product = np.dot(vec1, vec2)

        if normalize:
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                raise ValueError("One of the fingerprints has zero magnitude, cannot compute similarity.")

            dot_product = dot_product / (norm1 * norm2)

        similarity_metric = 0.5*(1 + dot_product)  # scale to [0, 1]
        return similarity_metric
