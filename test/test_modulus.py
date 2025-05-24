import numpy as np
import numpy.testing as npt
from scipy.optimize import Bounds

from lowprev_lsip.modulus import Domain, get_points


def test_points() -> None:
    npt.assert_allclose(
        get_points(3, Bounds([-1, 0], [2, 2])), np.array([[-1, 0], [-1, 1], [-1, 2]])
    )
