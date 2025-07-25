from collections.abc import Sequence
from dataclasses import dataclass

import numpy.typing as npt
import scipy.optimize


@dataclass
class LinearProgram:
    c: npt.NDArray
    A_ub: npt.NDArray
    b_ub: npt.NDArray
    bounds: Sequence[tuple[float | None, float | None]]


def solve_linear_program(lp: LinearProgram) -> tuple[float, npt.NDArray]:
    result = scipy.optimize.linprog(
        c=lp.c,
        A_ub=lp.A_ub,
        b_ub=lp.b_ub,
        bounds=lp.bounds,  # type: ignore
    )
    assert result.success, result.message
    return result.fun, result.x
