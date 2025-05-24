import logging
import math
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import numpy.typing as npt
import pytest
import scipy.optimize

_logger = logging.getLogger("lower_prevision.semi_infinite")


def oscillator(t: float) -> Callable[[Sequence[float]], float]:
    def _(x: npt.NDArray[float]) -> float:
        return np.sin(2 * math.pi * (t - x[1]) / x[0])

    return _


osc_bounds = scipy.optimize.Bounds(
    lb=[1, 0], ub=[2, 2]
)  # 1 <= x[0] <= 2, 0 <= x[1] <= 2
osc_constraints = [scipy.optimize.LinearConstraint(A=[[-1, 1]], lb=[0])]  # x[1] <= x[0]


def test_oscillator() -> None:
    assert oscillator(3)([1, 3]) == 0
    assert oscillator(3)([1, 5.5]) == pytest.approx(0)
    assert oscillator(3)([1, 6]) == pytest.approx(0)
    assert oscillator(1)([1, 1.25]) == pytest.approx(-1)
    assert oscillator(1)([1, 1.75]) == pytest.approx(1)
    assert oscillator(1)([2, 1.5]) == pytest.approx(-1)
    assert oscillator(1)([2, 2.5]) == pytest.approx(1)


def modulus_target(
    num_args: int, fun: Callable[[Sequence[float]], float]
) -> Callable[[Sequence[float]], float]:
    def _(xy: Sequence[float]) -> float:
        x = xy[:num_args]
        y = xy[num_args:]
        distance = math.fabs(fun(x) - fun(y))
        return -distance

    return _


def test_modulus_target() -> None:
    fun = modulus_target(2, lambda x: sum(x))
    assert fun([1, 2, 1, 2]) == 0
    assert fun([1, 2, 2, 1]) == 0
    assert fun([1, 2, 3, 4]) == -4


def target_constraint(constraint: scipy.optimize.LinearConstraint):
    zeros = np.zeros(constraint.A.shape)
    a = np.vstack([np.hstack([constraint.A, zeros]), np.hstack([zeros, constraint.A])])
    return scipy.optimize.LinearConstraint(
        A=a,
        lb=np.hstack([constraint.lb, constraint.lb]),
        ub=np.hstack([constraint.ub, constraint.ub]),
    )


def target_bounds(bounds: scipy.optimize.Bounds):
    return scipy.optimize.Bounds(
        lb=np.hstack([bounds.lb, bounds.lb]),
        ub=np.hstack([bounds.ub, bounds.ub]),
    )


def modulus_of_continuity(
    z: float,
    fun: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    x1: Sequence[float],
    bounds: scipy.optimize.Bounds,
    constraints: Sequence[scipy.optimize.LinearConstraint],
):
    target = modulus_target(len(x0), fun)
    z_constraint = scipy.optimize.LinearConstraint(
        A=[
            (
                [int(i == j) for j in range(len(x0))]
                + [-int(i == j) for j in range(len(x0))]
            )
            for i in range(len(x0))
        ],
        lb=[-z] * len(x0),
        ub=[z] * len(x0),
    )
    constraints = [z_constraint] + [
        target_constraint(constraint) for constraint in constraints
    ]
    for constraint in constraints:
        _logger.debug("constraint.A %s", constraint.A)
        _logger.debug("constraint.lb %s", constraint.lb)
        _logger.debug("constraint.ub %s", constraint.ub)
    bounds = target_bounds(bounds)
    _logger.debug("bounds.lb %s", bounds.lb)
    _logger.debug("bounds.ub %s", bounds.ub)
    return scipy.optimize.minimize(
        target,
        x0=np.hstack([x0, x1]),
        bounds=bounds,
        constraints=constraints,
    )


def modulus_of_continuity_2(
    z: float,
    fun: Callable[[Sequence[float]], float],
    grid: Sequence[Sequence[float]],
):
    def _values() -> Iterable[float]:
        for x0 in grid:
            for x10 in np.linspace(
                max(1.0, x0[0] - z / 2), min(2.0, x0[0] + z / 2), 10
            ):
                for x11 in np.linspace(
                    max(0.0, x0[1] - z / 2), min(2.0, x0[1] + z / 2), 10
                ):
                    if x11 <= x10:
                        yield math.fabs(fun(x0) - fun([x10, x11]))

    return max(_values())


def test_modulus_of_continuity() -> None:
    grid: Sequence[Sequence[float]] = [
        [x1, x2]
        for x1 in np.linspace(1, 2, 20)
        for x2 in np.linspace(0, 2, 20)
        if x2 <= x1
    ]
    for t in [3, 6, 9, 12]:
        for z in [0.01, 0.1]:
            for x0 in [[1.1, 0.1], [1.1, 0.9], [1.9, 1.9], [1.9, 0.1]]:
                x0_1 = np.array(x0) - np.array([z / 2, z / 2])
                x0_2 = np.array(x0) + np.array([z / 2, z / 2])
                result = modulus_of_continuity(
                    z, oscillator(t), x0_1, x0_2, osc_bounds, osc_constraints
                )
            print("OPT", t, z, -result.fun)
            mod_brute = modulus_of_continuity_2(z, oscillator(t), grid)
            print("BF ", t, z, mod_brute)
            mod_lip = z * 2 * math.pi * t
            print("LIP", t, z, mod_lip)
