from collections.abc import Callable, Iterable, Sequence
from typing import Protocol, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, brute, minimize

T = TypeVar("T")


def get_points(n: int, bounds: Bounds) -> Iterable[npt.NDArray]:
    points = np.linspace(bounds.lb[-1], bounds.ub[-1], n)
    if len(bounds.lb) == 1:
        return (np.array([x]) for x in points)
    else:
        return (
            np.append(x, y)
            for x in get_points(n, Bounds(bounds.lb[:-1], bounds.ub[:-1]))
            for y in points
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


def get_neighbourhood_bounds_from_max_norm(
    bounds: Bounds,
    x: npt.NDArray,
    z: float,
) -> Bounds:
    lb = np.max([bounds.lb, x - z], axis=0)
    ub = np.min([bounds.ub, x + z], axis=0)
    return Bounds(lb, ub)


def get_neighbourhood_from_max_norm(
    n: int,
    bounds: Bounds,
    z: float,
) -> Callable[[npt.NDArray], Iterable[npt.NDArray]]:
    def _(x: npt.NDArray) -> Iterable[npt.NDArray]:
        bounds2 = get_neighbourhood_bounds_from_max_norm(bounds, x, z)
        return get_points(n, bounds2)

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
    bounds: Bounds,
    n_points: int,
    n_neighbourhood: int,
    z: float,
) -> tuple[float, npt.NDArray, npt.NDArray]:
    points = list(get_points(n_points, bounds))
    return measure_modulus_of_continuity_from_neighbourhood(
        fun, get_neighbourhood_from_max_norm(n_neighbourhood, bounds, z), points
    )


class MinFun(Protocol):
    def __call__(
        self, fun: Callable[[npt.NDArray], float], bounds: Bounds
    ) -> float: ...


def modulus_of_continuity_at_2(
    min_fun: MinFun,
    fun: Callable[[npt.NDArray], float],
    bounds: Bounds,
    x: npt.NDArray,
    z: float,
) -> float:
    bounds2 = get_neighbourhood_bounds_from_max_norm(bounds, x, z)
    return -min_fun(lambda y: -abs(fun(x) - fun(y)), bounds2)


def measure_modulus_of_continuity_2(
    min_fun: MinFun,
    fun: Callable[[npt.NDArray], float],
    bounds: Bounds,
    z: float,
    min_fun_inner: MinFun | None = None,
) -> float:
    def fun2(x: npt.NDArray) -> float:
        return -modulus_of_continuity_at_2(
            min_fun if min_fun_inner is None else min_fun_inner, fun, bounds, x, z
        )

    return -min_fun(fun2, bounds)


def min_fun_brute(fun: Callable[[npt.NDArray], float], bounds: Bounds, ns=20) -> float:
    # note: scipy-stubs has too limited type checking for finish
    def finish(fun2, x0, args, **kwargs):
        return minimize(fun2, x0, args=args, bounds=bounds, **kwargs)

    _, val_opt, _, _ = brute(
        fun, tuple(zip(bounds.lb, bounds.ub)), Ns=ns, full_output=True, finish=finish
    )
    return val_opt
