import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
from scipy.optimize import Bounds

from lowprev_lsip.linear_program import (
    get_linear_program,
    solve_linear_program,
)
from lowprev_lsip.modulus import (
    lipschitz_constant,
    modulus_of_continuity,
)
from lowprev_lsip.optimize import min_fun_brute


def oscillator(t: float, x1: float, x2: float) -> float:
    return np.sin(2 * math.pi * (t - x1 * x2) / x1)


def oscillator_grad(t: float, x1: float, x2: float) -> tuple[float, float]:
    v = 2 * math.pi * np.cos(2 * math.pi * (t / x1 - x2))
    return -v * t / (x1**2), -v


osc_bounds = Bounds(lb=[1, 0], ub=[1.3, 0.1])  # 1 <= x[0] <= 1.3, 0 <= x[1] <= 0.1


def test_oscillator() -> None:
    assert oscillator(3, 1, 0) == pytest.approx(0)
    assert oscillator(3, 1, 0.5) == pytest.approx(0)
    assert oscillator(3, 1, 1) == pytest.approx(0)
    assert oscillator(1, 1, 1.25) == pytest.approx(-1)
    assert oscillator(1, 1, 1.75) == pytest.approx(1)
    assert oscillator(1, 2, 0.75) == pytest.approx(-1)
    assert oscillator(1, 2, 1.25) == pytest.approx(1)


def plot_oscillator(t: float, num: int, cmap: str) -> None:
    x = np.linspace(osc_bounds.lb[0], osc_bounds.ub[0], num)
    y = np.linspace(osc_bounds.lb[1], osc_bounds.ub[1], num)
    xx, yy = np.meshgrid(x, y)
    fun = np.frompyfunc(lambda x1, x2: oscillator(t, x1, x2), 2, 1)
    zz = fun(xx, yy).astype(float)
    plt.contourf(xx, yy, zz, cmap=cmap)
    plt.colorbar()
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(f"$f_t(x_1,x_2)$ for $t={t}$")
    plt.show()


@pytest.mark.parametrize(
    "t,z,expected",
    [
        (1.8, 0.1, 1.47),
        (2, 0.15, 1.92),
        (2.2, 0.2, 2),
    ],
)
def test_modulus_of_continuity(t: float, z: float, expected: float) -> None:
    def fun(x: npt.NDArray) -> float:
        return oscillator(t, x[0], x[1])

    mod2 = modulus_of_continuity(fun, osc_bounds, z, min_fun=min_fun_brute)
    assert mod2 == pytest.approx(expected, abs=0.01)


def plot_for_modulus(t: float, zs: npt.NDArray) -> None:
    def fun(x: npt.NDArray) -> float:
        return oscillator(t, x[0], x[1])

    def fun_grad(x: npt.NDArray) -> npt.NDArray:
        return np.array(oscillator_grad(t, x[0], x[1]))

    mods = [
        modulus_of_continuity(fun, osc_bounds, z, min_fun=min_fun_brute) for z in zs
    ]
    lip = lipschitz_constant(fun_grad, osc_bounds, min_fun=min_fun_brute)
    lips = [lip * z for z in zs]
    plt.ylim(0, 2.1)
    plt.plot(zs, mods, "C0", linestyle="-", label=r"$\xi_{f_t(X_1,X_2)}(z)$")
    plt.plot(zs, lips, "C1", linestyle="--", label=r"$z M_{f_t(X_1,X_2)}$")
    plt.legend()
    plt.title(f"Modulus of continuity of $f_t(X_1,X_2)$ for $t={t}$")
    plt.show()


def get_osc_lin_prog(t: float, num: int) -> tuple[float, float]:
    def y(omega: npt.NDArray) -> float:
        return oscillator(t, omega[0], omega[1])

    def y_neg(omega: npt.NDArray) -> float:
        return -oscillator(t, omega[0], omega[1])

    def x1(omega: npt.NDArray) -> float:
        return omega[0]

    def x2(omega: npt.NDArray) -> float:
        return omega[1]

    grid = np.meshgrid(
        *[np.linspace(lb, ub, num) for lb, ub in zip(osc_bounds.lb, osc_bounds.ub)]
    )
    points = np.vstack([coord.ravel() for coord in grid]).T
    low_prev = [(x1, 1.1, 1.2), (x2, 0.05, 0.07)]
    lp1 = get_linear_program(y, low_prev, points)
    lp2 = get_linear_program(y_neg, low_prev, points)
    return solve_linear_program(lp1), -solve_linear_program(lp2)


if __name__ == "__main__":
    for t in np.linspace(2, 5, 20):
        print(t, get_osc_lin_prog(t=t, num=100))
    plot_oscillator(t=2, num=30, cmap="plasma")
    plot_oscillator(t=5, num=200, cmap="plasma")
    plot_for_modulus(t=2, zs=np.linspace(0, 0.1, 10))
    plot_for_modulus(t=5, zs=np.linspace(0, 0.1, 30))
