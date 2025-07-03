from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.optimize


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
