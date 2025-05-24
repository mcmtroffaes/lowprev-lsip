from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint

T = TypeVar("T")


@dataclass
class Domain:
    bounds: Bounds
    linear_constraint: LinearConstraint | None = None


def get_points(n: int, bounds: Bounds) -> npt.NDArray:
    points = np.linspace(bounds.lb[-1], bounds.ub[-1], n)
    if len(bounds.lb) == 1:
        return points
    else:
        return np.array([
            np.append(x, y)
            for x in get_points(n, Bounds(bounds.lb[:-1], bounds.ub[:-1]))
            for y in points
            ]
        )


def get_points_with_linear_constraint(
    points: Iterable[npt.NDArray], linear_constraint: LinearConstraint
) -> Iterable[npt.NDArray]:
    for x in points:
        if linear_constraint.residual(x) >= 0:
            yield x


def get_neighbourhood_at(
    metric: Callable[[T, T], float], points: Iterable[T], x: T, z: float
) -> Iterable[T]:
    return (y for y in points if metric(x, y) <= z)


def get_max_norm_neighbourhood_at(
    n: int, x: npt.NDArray, z: float
) -> Iterable[npt.NDArray]:
    return get_points(n, Bounds(x - 0.5 * z, x + 0.5 * z))


def modulus_of_continuity_at(
    x: T,
    fun: Callable[[T], float],
    neighbourhood: Sequence[T],
) -> float:
    return max(abs(fun(x) - fun(y)) for y in neighbourhood)


def measure_modulus_of_continuity_1(
    metric: Callable[[T, T], float],
    points: Sequence[T],
    fun: Callable[[T], float],
    z: float,
) -> float:
    """Calculate modulus of continuity using a domain.
    This function implements the definition directly.
    Very slow (N^2 evaluations if the domain has N points).
    """
    return max(
        modulus_of_continuity_at(x, fun, get_neighbourhood_at(metric, points, x, z))
        for x in points
    )
