import numpy as np
from pygments.lexers.qvt import QVToLexer


def convert_to_spherical_coordinates(points: np.array):
    """
    Convert a set of points in 3D space to spherical coordinates (theta, phi).
    The first point is assumed to be the origin.

    Parameters:
    ----------
    points : np.array
        An array of shape (n, 3) where n is the number of points, and each point is represented by its (x, y, z) coordinates.
        The first point is considered the origin.

    Returns:
    ----------
    theta : np.array
        An array of angles from the z-axis for each point, in radians (length, n-1)
    phi : np.array
        An array of angles in the xy-plane for each point, in radians (length, n-1)
    r : np.array
        An array of distances from the origin for each point, in the same order as theta and phi (length, n-1)
    """

    # shift all points relative to the first point
    # then remove the first point from the array

    points_relative = points - points[0]
    points_relative = points_relative[1:]  # remove the first point (the origin)

    # calculate spherical coordinates
    # r = sqrt(x^2 + y^2 + z^2)
    r = np.linalg.norm(points_relative, axis=1)

    # theta = arccos(z / r), range [0 to pi]
    theta = np.arccos(points_relative[:, 2] / r)  # angle from z-axis

    # phi = arctan(y / x)
    phi = np.arctan2(points_relative[:, 1], points_relative[:, 0])  # angle in xy-plane
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)  # ensure phi is in [0, 2*pi]

    return theta, phi, r


def calculate_Q(theta: np.array, phi: np.array, l: int, include_w: bool = True):
    """
    Calculate invariant Q for a given frequency (l) and spherical coordinates (theta, phi).

    Parameters:
    ----------
    theta : np.array
        An array of angles from the z-axis for each point, in radians.
    phi : np.array
        An array of angles in the xy-plane for each point, in radians.
    l : int
        The degree of the spherical harmonics to calculate. Must be a non-negative integer.

    Returns:
    ----------

    """

    from scipy.special import sph_harm_y

    """
    General algorithm: 
    
    For a given frequency, l, we calculate q_l from the spherical harmonics Y_{l,m} for each point (theta_i, phi_i), 
    where m ranges from -l to l, normalizing by the number of points, N:
    
    q_l = (1/N) sum_{j}^{N} sum_{m=-l}^{l} Y_{l,m}(theta_j, phi_j)
    
    To get the rotationally invariant descriptor, we calculate the magnitude of the spherical harmonics:
    Q_l = sqrt((4 * pi / (2l + 1)) * (q_l^2))
    
    """

    # first initialize an array to hold the spherical harmonics.
    # The shape of this complex array is (2l + 1, N), where N is the number of points.
    # Since we are u sing numpy, we will operate on all points at once, rather than iterating over them, hence the 2D array.

    q_m_shell = np.zeros((2 * l + 1, theta.shape[0]), dtype=np.complex128)

    # ylm_shell_real = np.zeros(((2 * l + 1), theta.shape[0]), dtype=np.float64)
    # ylm_shell_imag = np.zeros(((2 * l + 1), theta.shape[0]), dtype=np.float64)

    for m in range(-l, l + 1):
        # Calculate the spherical harmonics using scipy's sph_harm_y function
        ylm = sph_harm_y(l, m, theta, phi)

        # update the q_m_shell array with the spherical harmonics values
        # m+l just shifts the index to account for negative m values
        q_m_shell[m + l] = q_m_shell[m + l] + ylm

        # ylm_shell_real[m + l] = ylm_shell_real[m + l] + ylm.real
        # ylm_shell_imag[m + l] = ylm_shell_imag[m + l] + ylm.imag
        # Accumulate the real and imaginary parts

    # for each m, we will sum the Ylm values across all points, and then normalize by the number of points
    # i.e., we calculate the mean of the values in each row of the q_m_shell array
    q_m_shell = np.mean(q_m_shell, axis=1)

    # ylm_shell_real = np.mean(ylm_shell_real, axis=1)
    # ylm_shell_imag = np.mean(ylm_shell_imag, axis=1)

    # next we will sum the squares of the real and imaginary parts q_m_shell
    # qlm_sq = np.sum(ylm_shell_real**2) + np.sum(ylm_shell_imag**2)
    qm_sq = np.sum(q_m_shell.real**2) + np.sum(q_m_shell.imag**2)

    # calculate the magnitude of to get the rotationally invariant descriptor for the given frequency l, Q_l
    Q_l = np.sqrt(4.0 * np.pi / (2 * l + 1) * qm_sq)

    return Q_l
