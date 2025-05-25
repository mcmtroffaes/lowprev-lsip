import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
from scipy.optimize import Bounds

from lowprev_lsip.modulus import (
    measure_modulus_of_continuity_2,
    measure_modulus_of_continuity_from_max_norm,
    min_fun_brute,
)


def oscillator(t: float, x1: float, x2: float) -> float:
    return np.sin(2 * math.pi * (t - x1 * x2) / x1)


osc_bounds = Bounds(lb=[1, 0], ub=[2, 1])  # 1 <= x[0] <= 2, 0 <= x[1] <= 1


def test_oscillator() -> None:
    assert oscillator(3, 1, 0) == pytest.approx(0)
    assert oscillator(3, 1, 0.5) == pytest.approx(0)
    assert oscillator(3, 1, 1) == pytest.approx(0)
    assert oscillator(1, 1, 1.25) == pytest.approx(-1)
    assert oscillator(1, 1, 1.75) == pytest.approx(1)
    assert oscillator(1, 2, 0.75) == pytest.approx(-1)
    assert oscillator(1, 2, 1.25) == pytest.approx(1)


def plot_oscillator(t: float) -> None:
    x = np.linspace(osc_bounds.lb[0], osc_bounds.ub[0], 10)
    y = np.linspace(osc_bounds.lb[1], osc_bounds.ub[1], 10)
    xx, yy = np.meshgrid(x, y)
    fun = np.frompyfunc(lambda x1, x2: oscillator(t, x1, x2), 2, 1)
    zz = np.sin(xx, yy)
    print(zz)
    plt.figure()
    plt.contourf(xx, yy, zz, np.linspace(-1, 1, 5))
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
def test_modulus_of_continuity_1(t: float, z: float, expected: float) -> None:
    def fun(x: npt.NDArray) -> float:
        return oscillator(t, x[0], x[1])

    mod, x0, x1 = measure_modulus_of_continuity_from_max_norm(
        fun, osc_bounds, 50, 10, z
    )
    assert mod == pytest.approx(expected, abs=0.01)
    assert abs(fun(x0) - fun(x1)) == pytest.approx(mod)
    assert np.max(np.abs(x0 - x1)) == pytest.approx(z)
    mod2 = measure_modulus_of_continuity_2(min_fun_brute, fun, osc_bounds, z)
    assert mod2 == pytest.approx(expected, abs=0.01)


def plot_for_modulus() -> None:
    t = 1.5

    def fun(x: npt.NDArray) -> float:
        return oscillator(t, x[0], x[1])

    zs = np.linspace(0, 0.1, 10)
    mods = [
        measure_modulus_of_continuity_from_max_norm(fun, osc_bounds, 50, 10, z)[2]
        for z in zs
    ]
    lips = [2 * np.pi * z * max(t, 1) for z in zs]
    # TODO complete


if __name__ == "__main__":
    plot_oscillator(1.5)
