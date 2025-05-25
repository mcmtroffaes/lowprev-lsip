from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import pytest
from scipy.optimize import Bounds

from lowprev_lsip.modulus import (
    get_neighbourhood_from_max_norm,
    get_points,
    measure_modulus_of_continuity_2,
    measure_modulus_of_continuity_from_max_norm,
    measure_modulus_of_continuity_from_metric,
    minimize_brute,
)


def assert_points_equal(
    actual: Sequence[npt.NDArray], expected: Sequence[Sequence[float]]
) -> None:
    assert len(actual) == len(expected)
    for x, y in zip(actual, expected):
        assert x == pytest.approx(y)


def test_get_points_1() -> None:
    assert_points_equal(
        list(get_points(3, Bounds([-1, 0], [2, 2]))),
        [
            [-1, 0],
            [-1, 1],
            [-1, 2],
            [0.5, 0],
            [0.5, 1],
            [0.5, 2],
            [2, 0],
            [2, 1],
            [2, 2],
        ],
    )


def test_get_neighbourhood_from_max_norm() -> None:
    # -1 <= x <= 2, 0 <= y <= 2
    bounds = Bounds([-1, 0], [2, 2])
    xs = list(get_neighbourhood_from_max_norm(3, bounds, 0.2)(np.array([0.1, 0.1])))
    assert_points_equal(
        xs,
        [
            [-0.1, 0.0],
            [-0.1, 0.15],
            [-0.1, 0.3],
            [0.1, 0.0],
            [0.1, 0.15],
            [0.1, 0.3],
            [0.3, 0.0],
            [0.3, 0.15],
            [0.3, 0.3],
        ],
    )


@pytest.mark.parametrize(
    "fun,z,expected",
    [
        (lambda x: x, 0.1, 0.1),
        (lambda x: x**2, 0.2, 1 - 0.8**2),
        (lambda x: -2 * x, 0.3, 0.6),
    ],
)
def test_modulus_of_continuity_from_metric(
    fun: Callable[[float], float], z: float, expected: float
) -> None:
    points = [x for x in np.linspace(0, 1, 100)]
    mod, x0, x1 = measure_modulus_of_continuity_from_metric(
        fun, lambda x, y: abs(x - y), points, z
    )
    assert mod == pytest.approx(expected, rel=0.1)  # large error due to poor grid
    assert abs(fun(x0) - fun(x1)) == pytest.approx(mod)  # should be exact
    assert np.max(np.abs(x0 - x1)) == pytest.approx(z, abs=0.01)  # 100 points, so 1/100


@pytest.mark.parametrize(
    "fun,z,expected",
    [
        (lambda x: x[0], 0.1, 0.1),
        (lambda x: x[0] ** 2, 0.2, 1 - 0.8**2),
        (lambda x: -2 * x[0], 0.3, 0.6),
    ],
)
def test_modulus_of_continuity_from_max_norm(
    fun: Callable[[npt.NDArray], float], z: float, expected: float
) -> None:
    bounds = Bounds(0, 1)
    mod, x0, x1 = measure_modulus_of_continuity_from_max_norm(fun, bounds, 100, 10, z)
    assert mod == pytest.approx(expected)
    assert abs(fun(x0) - fun(x1)) == pytest.approx(mod)
    assert np.max(np.abs(x0 - x1)) == pytest.approx(z)


@pytest.mark.parametrize(
    "fun,z,expected",
    [
        (lambda x: x[0], 0.1, 0.1),
        (lambda x: x[0] ** 2, 0.2, 1 - 0.8**2),
        (lambda x: -2 * x[0], 0.3, 0.6),
    ],
)
def test_modulus_of_continuity_2(
    fun: Callable[[npt.NDArray], float], z: float, expected: float
) -> None:
    bounds = Bounds(0, 1)
    mod = measure_modulus_of_continuity_2(minimize_brute, fun, bounds, z)
    assert mod == pytest.approx(expected)
