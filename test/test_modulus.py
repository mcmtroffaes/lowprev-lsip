import functools
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
from scipy.optimize import Bounds

from lowprev_lsip.modulus import (
    get_neighbourhood_bounds,
    lipschitz_constant,
    modulus_of_continuity,
    modulus_of_continuity_slow,
)
from lowprev_lsip.optimize import MinFun, min_fun_brute

# fast min_fun_brute
min_fun_brute_2: Callable[[Bounds], MinFun] = functools.partial(min_fun_brute, ns=2)


def test_get_neighbourhood_bounds() -> None:
    # -1 <= x <= 2, 0 <= y <= 2
    bounds = Bounds([-1, 0], [2, 2])
    bounds2 = get_neighbourhood_bounds(bounds, np.array([0.1, 0.1]), 0.2)
    assert bounds2.lb == pytest.approx([-0.1, 0])
    assert bounds2.ub == pytest.approx([0.3, 0.3])


@pytest.mark.parametrize(
    "fun,z,expected",
    [
        (lambda x: x[0], 0.1, 0.1),
        (lambda x: x[0] ** 2, 0.2, 1 - 0.8**2),
        (lambda x: -2 * x[0], 0.3, 0.6),
    ],
)
def test_modulus_of_continuity(
    fun: Callable[[npt.NDArray], float], z: float, expected: float
) -> None:
    mod, x0, x1 = modulus_of_continuity_slow(
        fun, [np.array([x]) for x in np.linspace(0, 1, 100)], z
    )
    assert mod == pytest.approx(expected, rel=0.1)  # large error due to poor grid
    assert abs(fun(x0) - fun(x1)) == pytest.approx(mod)  # should be exact
    assert abs(x0 - x1) == pytest.approx(z, abs=0.01)  # 100 points, so 1/100
    mod = modulus_of_continuity(
        fun, Bounds(0, 1), z, min_fun=min_fun_brute_2, min_fun_inner=min_fun_brute_2
    )
    assert mod == pytest.approx(expected)
    mod2 = modulus_of_continuity(
        fun, Bounds(0, 1), z, min_fun=min_fun_brute_2, min_fun_inner=min_fun_brute_2
    )
    assert mod2 == pytest.approx(expected)


@pytest.mark.parametrize(
    "fun,fun_grad,z,expected",
    [
        (lambda x: x[0] + 2 * x[1], lambda x: np.array([1, 2]), 0.1, 0.3),
        (lambda x: -3 * x[0] + x[1], lambda x: np.array([-3, 1]), 0.2, 0.8),
    ],
)
def test_modulus_of_continuity_2(
    fun: Callable[[npt.NDArray], float],
    fun_grad: Callable[[npt.NDArray], npt.NDArray],
    z: float,
    expected: float,
) -> None:
    bounds = Bounds([0, 0], [1, 1])
    mod = modulus_of_continuity(fun, bounds, z, min_fun_brute_2, min_fun_brute_2)
    assert mod == pytest.approx(expected)
    lip = lipschitz_constant(fun_grad, min_fun_brute(bounds))
    assert mod == pytest.approx(lip * z)


@pytest.mark.parametrize(
    "fun,fun_grad,z,exp_mod,exp_lip",
    [
        (
            lambda x: x[0] ** 2 + x[1] ** 2,
            lambda x: 2 * x,
            0.1,
            2 - 0.9**2 - 0.9**2,
            2 + 2,
        ),
    ],
)
def test_modulus_of_continuity_3(
    fun: Callable[[npt.NDArray], float],
    fun_grad: Callable[[npt.NDArray], npt.NDArray],
    z: float,
    exp_mod: float,
    exp_lip: float,
) -> None:
    bounds = Bounds([0, 0], [1, 1])
    mod = modulus_of_continuity(fun, bounds, z, min_fun_brute, min_fun_brute_2)
    assert mod == pytest.approx(exp_mod)
    lip = lipschitz_constant(fun_grad, min_fun_brute(bounds))
    assert lip == pytest.approx(exp_lip)
    # mod <= lip * z
    assert min([0, lip * z - mod]) == pytest.approx(0)
