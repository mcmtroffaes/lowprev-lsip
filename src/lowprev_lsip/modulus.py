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


def _get_points(n: int, bounds: Bounds) -> Iterable[npt.NDArray]:
    points = np.linspace(bounds.lb[-1], bounds.ub[-1], n)
    if len(bounds.lb) == 1:
        return (np.array([x]) for x in points)
    else:
        return (
            np.append(x, y)
            for x in _get_points(n, Bounds(bounds.lb[:-1], bounds.ub[:-1]))
            for y in points
        )


def get_points(n: int, domain: Domain, tol=1e-6) -> Iterable[npt.NDArray]:
    def is_feasible(x: npt.NDArray):
        assert domain.linear_constraint is not None
        sl: npt.NDArray
        sb: npt.NDArray
        sl, sb = domain.linear_constraint.residual(x)
        return np.min([sl, sb]) >= -tol

    points = _get_points(n, domain.bounds)
    return (
        points
        if domain.linear_constraint is None
        else (x for x in points if is_feasible(x))
    )


def get_neighbourhood_from_metric(
    metric: Callable[[T, T], float],
    points: Sequence[T],
    z: float,
    tol=1e-6,
) -> Callable[[T], Iterable[T]]:
    def _(x: T) -> Iterable[T]:
        return (y for y in points if metric(x, y) <= z + tol)

    return _


def get_neighbourhood_from_max_norm(
    n: int,
    domain: Domain,
    z: float,
    tol=1e-6,
) -> Callable[[npt.NDArray], Iterable[npt.NDArray]]:
    def _(x: npt.NDArray) -> Iterable[npt.NDArray]:
        lb = np.max([domain.bounds.lb, x - z], axis=0)
        ub = np.min([domain.bounds.ub, x + z], axis=0)
        neighbourhood_domain = Domain(Bounds(lb, ub), domain.linear_constraint)
        return get_points(n, neighbourhood_domain, tol=tol)

    return _


def modulus_of_continuity_at(
    fun: Callable[[T], float],
    neighbourhood: Iterable[T],
    x: T,
) -> tuple[float, T, T]:
    """Calculate maximum distance between *fun* of *x*
    and *fun* of all values in *neighbourhood*.
    """
    return max(
        ((abs(fun(x) - fun(y)), x, y) for y in neighbourhood), key=lambda x: x[0]
    )


def measure_modulus_of_continuity_from_neighbourhood(
    fun: Callable[[T], float],
    neighbourhood: Callable[[T], Iterable[T]],
    points: Sequence[T],
) -> tuple[float, T, T]:
    """Calculate modulus of continuity from given points and neighbourhoods."""
    return max(modulus_of_continuity_at(fun, neighbourhood(x), x) for x in points)


def measure_modulus_of_continuity_from_metric(
    fun: Callable[[T], float],
    metric: Callable[[T, T], float],
    points: Sequence[T],
    z: float,
    tol=1e-6,
) -> tuple[float, T, T]:
    """Calculate modulus of continuity using a domain.
    This function implements the definition directly.
    Very slow (N^2 evaluations if the domain has N points).
    """
    return measure_modulus_of_continuity_from_neighbourhood(
        fun, get_neighbourhood_from_metric(metric, points, z, tol=tol), points
    )


def measure_modulus_of_continuity_from_max_norm(
    fun: Callable[[npt.NDArray], float],
    domain: Domain,
    n_points: int,
    n_neighbourhood: int,
    z: float,
    tol=1e-6,
) -> tuple[float, npt.NDArray, npt.NDArray]:
    points = list(get_points(n_points, domain, tol))
    return measure_modulus_of_continuity_from_neighbourhood(
        fun, get_neighbourhood_from_max_norm(n_neighbourhood, domain, z), points
    )
