from collections.abc import Callable
from typing import Protocol

import numpy.typing as npt
from scipy.optimize import Bounds, brute, minimize


class MinFun(Protocol):
    def __call__(
        self, fun: Callable[[npt.NDArray], float], bounds: Bounds
    ) -> float: ...


def min_fun_minimize(fun: Callable[[npt.NDArray], float], bounds: Bounds) -> float:
    """Wrapper around :func:`scipy.optimize.minimize`.
    Suitable for convex functions.
    """
    return minimize(fun, 0.5 * (bounds.lb + bounds.ub), bounds=bounds).fun


def min_fun_brute(fun: Callable[[npt.NDArray], float], bounds: Bounds, ns=20) -> float:
    """Wrapper around :func:`scipy.optimize.brute`.
    Finishes with `scipy.optimize.minimize` to respect the bounds.
    Suitable for non-linear functions.
    """

    # note: scipy-stubs has too limited type checking for finish
    def finish(fun2, x0, args, **kwargs):
        return minimize(fun2, x0, args=args, bounds=bounds, **kwargs)

    _, val_opt, _, _ = brute(
        fun, tuple(zip(bounds.lb, bounds.ub)), Ns=ns, full_output=True, finish=finish
    )
    return val_opt
