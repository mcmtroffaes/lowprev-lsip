from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds

from lowprev_lsip.linear_program import LinearProgram, solve_linear_program
from lowprev_lsip.optimize import MinFun, max_fun

Gamble = Callable[[npt.NDArray], float]


def get_low_prev_linear_program(
    y: Gamble,
    low_prev: Sequence[tuple[Gamble, float]],
    points: Sequence[npt.NDArray],
) -> LinearProgram:
    # minimize -alpha
    return LinearProgram(
        c=np.array([0] * len(low_prev) + [-1]),
        A_ub=np.array(
            [([x(omega) - lp for x, lp in low_prev] + [1]) for omega in points]
        ),
        b_ub=np.array([y(omega) for omega in points]),
        bounds=[(0, None)] * len(low_prev) + [(None, None)],
    )


def get_conjugate_gamble(y: Gamble) -> Gamble:
    def y_neg(t: npt.NDArray) -> float:
        return -y(t)

    return y_neg


def get_conjugate_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    return -bounds[1], -bounds[0]


@dataclass
class NaturalExtensionResult1:
    lambda_: npt.NDArray
    alpha: float


def max_discrepancy(
    y: Gamble,
    low_prev: Sequence[tuple[Gamble, float]],
    result: NaturalExtensionResult1,
    bounds: Bounds,
    min_fun: MinFun,
) -> tuple[npt.NDArray, float]:
    def h(t: npt.NDArray) -> float:
        return (
            -y(t)
            + result.alpha
            + sum([lam * (x(t) - lp) for lam, (x, lp) in zip(result.lambda_, low_prev)])
        )

    return max_fun(min_fun, h, bounds)


def solve_natural_extension_1(
    y: Gamble, low_prev: Sequence[tuple[Gamble, float]], points: Sequence[npt.NDArray]
) -> NaturalExtensionResult1:
    lin_prog = get_low_prev_linear_program(y, low_prev, points)
    _, lambda_alpha = solve_linear_program(lin_prog)
    assert lambda_alpha.shape == (1 + len(low_prev),)
    alpha = lambda_alpha[-1]
    assert isinstance(alpha, float)
    return NaturalExtensionResult1(lambda_alpha[: len(low_prev)], alpha)


def solve_conjugate_natural_extension_1(
    y: Gamble, low_prev: Sequence[tuple[Gamble, float]], points: Sequence[npt.NDArray]
) -> NaturalExtensionResult1:
    result = solve_natural_extension_1(get_conjugate_gamble(y), low_prev, points)
    return NaturalExtensionResult1(result.lambda_, -result.alpha)


@dataclass
class NaturalExtensionResult2:
    lambda_: npt.NDArray
    alpha_bounds: tuple[float, float]
    points: Sequence[npt.NDArray]


def solve_natural_extension_2(
    y: Gamble,
    low_prev: Sequence[tuple[Gamble, float]],
    initial_points: Sequence[npt.NDArray],
    bounds: Bounds,
    min_fun: MinFun,
    error: float,
) -> NaturalExtensionResult2:
    points = list(initial_points)  # U_0
    while True:
        result = solve_natural_extension_1(y, low_prev, points)
        t_next, delta = max_discrepancy(y, low_prev, result, bounds, min_fun)
        delta_tilde = max(0.0, delta)
        if delta_tilde < error:
            return NaturalExtensionResult2(
                lambda_=result.lambda_,
                alpha_bounds=(result.alpha - delta_tilde, result.alpha),
                points=points,
            )
        points.append(t_next)


def solve_conjugate_natural_extension_2(
    y: Gamble,
    low_prev: Sequence[tuple[Gamble, float]],
    initial_points: Sequence[npt.NDArray],
    bounds: Bounds,
    min_fun: MinFun,
    error: float,
) -> NaturalExtensionResult2:
    result = solve_natural_extension_2(
        get_conjugate_gamble(y), low_prev, initial_points, bounds, min_fun, error
    )
    return NaturalExtensionResult2(
        lambda_=result.lambda_,
        alpha_bounds=get_conjugate_bounds(result.alpha_bounds),
        points=result.points,
    )
