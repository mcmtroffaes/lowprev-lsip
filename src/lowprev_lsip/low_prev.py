import time
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


@dataclass
class NaturalExtensionResult1:
    # note: true solution lies between alpha - delta_tilde and alpha
    lambda_: npt.NDArray
    alpha: float
    t_next: npt.NDArray
    delta_tilde: float
    time: float


def max_discrepancy(
    y: Gamble,
    low_prev: Sequence[tuple[Gamble, float]],
    lambda_: npt.NDArray,
    alpha: float,
    bounds: Bounds,
    min_fun: MinFun,
) -> tuple[npt.NDArray, float]:
    def h(t: npt.NDArray) -> float:
        return (
            -y(t)
            + alpha
            + sum([lam * (x(t) - lp) for lam, (x, lp) in zip(lambda_, low_prev)])
        )

    return max_fun(min_fun, h, bounds)


def solve_natural_extension_1(
    y: Gamble,
    low_prev: Sequence[tuple[Gamble, float]],
    points: Sequence[npt.NDArray],
    bounds: Bounds,
    min_fun: MinFun,
) -> NaturalExtensionResult1:
    start = time.time()
    lin_prog = get_low_prev_linear_program(y, low_prev, points)
    _, lambda_alpha = solve_linear_program(lin_prog)
    assert lambda_alpha.shape == (1 + len(low_prev),)
    lambda_ = lambda_alpha[: len(low_prev)]
    alpha = lambda_alpha[-1]
    assert isinstance(alpha, float)
    t_next, delta = max_discrepancy(y, low_prev, lambda_, alpha, bounds, min_fun)
    delta_tilde = max(0.0, delta)
    return NaturalExtensionResult1(
        lambda_=lambda_,
        alpha=alpha,
        t_next=t_next,
        delta_tilde=delta_tilde,
        time=time.time() - start,
    )


@dataclass
class NaturalExtensionResult2(NaturalExtensionResult1):
    all_points: Sequence[npt.NDArray]
    all_results: Sequence[NaturalExtensionResult1]


def solve_natural_extension_2(
    y: Gamble,
    low_prev: Sequence[tuple[Gamble, float]],
    initial_points: Sequence[npt.NDArray],
    bounds: Bounds,
    min_fun: MinFun,
    tolerance: float,
) -> NaturalExtensionResult2:
    start = time.time()
    points = list(initial_points)  # U_0
    all_results: list[NaturalExtensionResult1] = []
    while True:
        result = solve_natural_extension_1(y, low_prev, points, bounds, min_fun)
        all_results.append(result)
        if result.delta_tilde < tolerance:
            return NaturalExtensionResult2(
                lambda_=result.lambda_,
                alpha=result.alpha,
                t_next=result.t_next,
                delta_tilde=result.delta_tilde,
                time=time.time() - start,
                all_points=points,
                all_results=all_results,
            )
        points.append(result.t_next)
