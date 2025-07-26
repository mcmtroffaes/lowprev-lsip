from collections.abc import Callable
from typing import Protocol

import numpy.typing as npt
from scipy.optimize import (
    Bounds,
    OptimizeResult,
    brute,
    differential_evolution,
    direct,
    dual_annealing,
    minimize,
    shgo,
)


class MinFun(Protocol):
    def __call__(
        self, fun: Callable[[npt.NDArray], float]
    ) -> tuple[float, npt.NDArray]: ...


class _SciPyMinFun(Protocol):
    def __call__(
        self, fun: Callable[[npt.NDArray], float], bounds: Bounds
    ) -> OptimizeResult: ...


def max_fun(
    min_fun: MinFun, fun: Callable[[npt.NDArray], float]
) -> tuple[float, npt.NDArray]:
    f_star, x_star = min_fun(lambda x: -fun(x))
    return -f_star, x_star


def min_fun_minimize(bounds: Bounds, x0: npt.NDArray) -> MinFun:
    """Wrapper around :func:`scipy.optimize.minimize`.
    Suitable for convex functions.
    """

    def _(fun: Callable[[npt.NDArray], float]) -> tuple[float, npt.NDArray]:
        result = minimize(fun, x0, bounds=bounds)
        return result.fun, result.x

    return _


def min_fun_brute(bounds: Bounds, ns=20) -> MinFun:
    """Wrapper around :func:`scipy.optimize.brute`.
    Finishes with `scipy.optimize.minimize` to respect the bounds.
    Suitable for non-linear functions.
    """
    assert ns >= 2  # brute fails otherwise

    def _(fun: Callable[[npt.NDArray], float]) -> tuple[float, npt.NDArray]:
        # note: scipy-stubs has too limited type checking for finish
        def finish(fun2, x0, args, **kwargs):
            return minimize(fun2, x0, args=args, bounds=bounds, **kwargs)

        x_star, fun_star, _, _ = brute(
            fun,
            tuple(zip(bounds.lb, bounds.ub)),
            Ns=ns,
            full_output=True,
            finish=finish,
        )
        return fun_star, x_star

    return _


def _min_fun_wrap(scipy_min_fun: _SciPyMinFun) -> Callable[[Bounds], MinFun]:
    def min_fun(bounds: Bounds) -> MinFun:
        def _(fun: Callable[[npt.NDArray], float]) -> tuple[float, npt.NDArray]:
            result = scipy_min_fun(fun, bounds)
            return result.fun, result.x

        return _

    return min_fun


min_fun_differential_evolution = _min_fun_wrap(
    lambda fun, bounds: differential_evolution(fun, bounds)
)
min_fun_shgo = _min_fun_wrap(lambda fun, bounds: shgo(fun, bounds))
min_fun_dual_annealing = _min_fun_wrap(lambda fun, bounds: dual_annealing(fun, bounds))
min_fun_direct = _min_fun_wrap(lambda fun, bounds: direct(fun, bounds))
