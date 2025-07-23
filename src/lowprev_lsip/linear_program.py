from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.optimize
from scipy.optimize import Bounds

from lowprev_lsip.optimize import MinFun, max_fun


@dataclass
class LinearProgram:
    c: npt.NDArray
    A_ub: npt.NDArray
    b_ub: npt.NDArray
    bounds: Sequence[tuple[float | None, float | None]]


Gamble = Callable[[npt.NDArray], float]


def get_linear_program(
    y: Gamble,
    low_prev: Sequence[tuple[Gamble, float, float]],
    points: Sequence[npt.NDArray],
) -> LinearProgram:
    # minimize -alpha
    return LinearProgram(
        c=np.array([0] * (2 * len(low_prev)) + [-1]),
        A_ub=np.array(
            [
                (
                    [x(omega) - lp for x, lp, up in low_prev]
                    + [up - x(omega) for x, lp, up in low_prev]
                    + [1]
                )
                for omega in points
            ]
        ),
        b_ub=np.array([y(omega) for omega in points]),
        bounds=[(0, None)] * (2 * len(low_prev)) + [(None, None)],
    )


def solve_linear_program(lp: LinearProgram) -> tuple[float, npt.NDArray]:
    result = scipy.optimize.linprog(
        c=lp.c,
        A_ub=lp.A_ub,
        b_ub=lp.b_ub,
        bounds=lp.bounds,  # type: ignore
    )
    assert result.success, result.message
    return result.fun, result.x


@dataclass
class NaturalExtensionResult:
    bounds: tuple[float, float]
    lambda_alpha: npt.NDArray
    points: Sequence[npt.NDArray]


def get_conjugate_gamble(y: Gamble) -> Gamble:
    def y_neg(t: npt.NDArray) -> float:
        return -y(t)

    return y_neg


def get_conjugate_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    return -bounds[1], -bounds[0]


def solve_natural_extension(
    y: Gamble,
    low_prev: Sequence[tuple[Gamble, float, float]],
    initial_points: Sequence[npt.NDArray],
    bounds: Bounds,
    min_fun: MinFun,
    error: float,
) -> NaturalExtensionResult:
    points = list(initial_points)  # U_0
    while True:
        lin_prog = get_linear_program(y, low_prev, points)
        _, lambda_alpha = solve_linear_program(lin_prog)
        assert len(lambda_alpha) == 1 + 2 * len(low_prev)
        lambda_lp = lambda_alpha[: len(low_prev)]
        lambda_up = lambda_alpha[len(low_prev) : 2 * len(low_prev)]
        alpha: float = lambda_alpha[-1]

        def h(t: npt.NDArray) -> float:
            return (
                -y(t)
                + alpha
                + sum(
                    [lam * (x(t) - lp) for lam, (x, lp, up) in zip(lambda_lp, low_prev)]
                )
                + sum(
                    [lam * (up - x(t)) for lam, (x, lp, up) in zip(lambda_up, low_prev)]
                )
            )

        t_next, delta = max_fun(min_fun, h, bounds)
        delta_tilde = max(0.0, delta)
        if delta_tilde < error:
            return NaturalExtensionResult(
                bounds=(alpha - delta_tilde, alpha),
                lambda_alpha=lambda_alpha,
                points=points,
            )
        points.append(t_next)


def solve_conjugate_natural_extension(
    y: Gamble,
    low_prev: Sequence[tuple[Gamble, float, float]],
    initial_points: Sequence[npt.NDArray],
    bounds: Bounds,
    min_fun: MinFun,
    error: float,
) -> NaturalExtensionResult:
    result = solve_natural_extension(
        get_conjugate_gamble(y), low_prev, initial_points, bounds, min_fun, error
    )
    return NaturalExtensionResult(
        bounds=get_conjugate_bounds(result.bounds),
        lambda_alpha=result.lambda_alpha,
        points=result.points,
    )
