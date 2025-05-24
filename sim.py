import logging
import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import pytest
from scipy.optimize import Bounds, LinearConstraint

from lowprev_lsip.modulus import (
    Domain,
    measure_modulus_of_continuity_from_max_norm,
)


def oscillator(t: float) -> Callable[[npt.NDArray], float]:
    def _(x: npt.NDArray) -> float:
        return np.sin(2 * math.pi * (t - x[1]) / x[0])

    return _


osc_domain = Domain(
    Bounds(lb=[1, 0], ub=[2, 2]),  # 1 <= x[0] <= 2, 0 <= x[1] <= 2
    LinearConstraint(A=[[-1, 1]], lb=[0]),  # x[1] <= x[0]
)


def test_oscillator() -> None:
    assert oscillator(3)(np.array([1, 3])) == 0
    assert oscillator(3)(np.array([1, 5.5])) == pytest.approx(0)
    assert oscillator(3)(np.array([1, 6])) == pytest.approx(0)
    assert oscillator(1)(np.array([1, 1.25])) == pytest.approx(-1)
    assert oscillator(1)(np.array([1, 1.75])) == pytest.approx(1)
    assert oscillator(1)(np.array([2, 1.5])) == pytest.approx(-1)
    assert oscillator(1)(np.array([2, 2.5])) == pytest.approx(1)


@pytest.mark.parametrize(
    "fun,z,expected",
    [
        (oscillator(1.8), 0.1, 0.87),
        (oscillator(2), 0.2, 1.52),
        (oscillator(2.2), 0.3, 1.93),
    ],
)
def test_modulus_of_continuity_1(
    fun: Callable[[npt.NDArray], float], z: float, expected: float
) -> None:
    mod, x0, x1 = measure_modulus_of_continuity_from_max_norm(
        fun, osc_domain, 50, 10, z
    )
    assert mod == pytest.approx(expected, abs=0.01)
    assert abs(fun(x0) - fun(x1)) == pytest.approx(mod)
    assert np.max(np.abs(x0 - x1)) == pytest.approx(z)
