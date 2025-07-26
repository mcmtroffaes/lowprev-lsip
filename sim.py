import functools
import logging
import math
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
from scipy.optimize import Bounds

from lowprev_lsip.low_prev import (
    Gamble,
    GambleGrad,
    NaturalExtensionResult,
    get_conjugate_gamble,
    solve_natural_extension_1,
    solve_natural_extension_2,
)
from lowprev_lsip.modulus import (
    lipschitz_constant,
    modulus_of_continuity,
)
from lowprev_lsip.optimize import MinFun, min_fun_brute, min_fun_minimize

# hard bounds
# 1 <= x1 <= 2
# pi/2 <= x2 <= pi
x1_lb = 1.0
x1_ub = 2.0
x2_lb = 0.5 * math.pi
x2_ub = math.pi

# lower and upper previsions
x1_lp = 1.4
x1_up = 1.5
x2_lp = 3.0
x2_up = 3.1


def oscillator(t: float, x1: float, x2: float) -> float:
    return np.sin(x2 + 2 * math.pi * t / x1)


def oscillator_grad(t: float, x1: float, x2: float) -> tuple[float, float]:
    v = np.cos(x2 + 2 * math.pi * t / x1)
    return -2 * math.pi * v * t / (x1**2), v


osc_bounds = Bounds(lb=[x1_lb, x2_lb], ub=[x1_ub, x2_ub])


def osc_x1(omega: npt.NDArray) -> float:
    return omega[0]


def osc_x2(omega: npt.NDArray) -> float:
    return omega[1]


osc_low_prev: Sequence[tuple[Gamble, float]] = [
    (osc_x1, x1_lp),
    (get_conjugate_gamble(osc_x1), -x1_up),
    (osc_x2, x2_lp),
    (get_conjugate_gamble(osc_x2), -x2_up),
]


def osc_y(t: float) -> Gamble:
    def _(omega: npt.NDArray) -> float:
        return oscillator(t, omega[0], omega[1])

    return _


def osc_y_grad(t: float) -> GambleGrad:
    def _(omega: npt.NDArray) -> npt.NDArray:
        return np.array(oscillator_grad(t, omega[0], omega[1]))

    return _


def test_oscillator() -> None:
    assert oscillator(3, 1, 0) == pytest.approx(0)
    assert oscillator(3, 1, -math.pi) == pytest.approx(0)
    assert oscillator(3, 1, -2 * math.pi) == pytest.approx(0)
    assert oscillator(1, 1, -2.5 * math.pi) == pytest.approx(-1)
    assert oscillator(1, 1, -3.5 * math.pi) == pytest.approx(1)
    assert oscillator(1, 2, -3.5 * math.pi) == pytest.approx(-1)
    assert oscillator(1, 2, -2.5 * math.pi) == pytest.approx(1)


def plot_oscillator(t: float, num: int, cmap: str) -> None:
    x = np.linspace(osc_bounds.lb[0], osc_bounds.ub[0], num)
    y = np.linspace(osc_bounds.lb[1], osc_bounds.ub[1], num)
    xx, yy = np.meshgrid(x, y)
    fun = np.frompyfunc(lambda x1, x2: oscillator(t, x1, x2), 2, 1)
    zz = fun(xx, yy).astype(float)
    plt.contourf(xx, yy, zz, cmap=cmap)
    plt.colorbar()
    plt.xlabel("$t_1$")
    plt.ylabel("$t_2$")
    plt.title(rf"$f_\tau(t_1,t_2)$ for $\tau={t}$")
    plt.show()


@pytest.mark.parametrize(
    "t,z,expected",
    [
        (1.8, 0.1, 1.07),
        (2, 0.15, 1.42),
        (2.2, 0.2, 1.90),
    ],
)
def test_modulus_of_continuity(t: float, z: float, expected: float) -> None:
    mod2 = modulus_of_continuity(osc_y(t), osc_bounds, z, min_fun_brute, min_fun_brute)
    assert mod2 == pytest.approx(expected, abs=0.01)


def plot_for_modulus(t: float, zs: npt.NDArray) -> None:
    mods = [
        modulus_of_continuity(
            osc_y(t),
            osc_bounds,
            z,
            min_fun_brute,
            functools.partial(min_fun_brute, ns=2),
        )
        for z in zs
    ]
    lip = lipschitz_constant(osc_y_grad(t), min_fun_brute(osc_bounds))
    lips = [lip * z for z in zs]
    plt.ylim(0, 2.1)
    plt.plot(zs, mods, "C0", linestyle="-", label=r"$\xi_{f_t(X_1,X_2)}(z)$")
    plt.plot(zs, lips, "C1", linestyle="--", label=r"$z M_{f_t(X_1,X_2)}$")
    plt.legend()
    plt.title(f"Modulus of continuity of $f_t(X_1,X_2)$ for $t={t}$")
    plt.show()


@dataclass
class SimulationResult:
    grid: Mapping[int, NaturalExtensionResult]
    semi: Mapping[float, Sequence[NaturalExtensionResult]]


def plot_alpha_bound(
    simulation: Mapping[float, SimulationResult], error: float
) -> None:
    ts: Sequence[float] = list(simulation.keys())
    inf_fs = np.array(
        [min_fun_brute(osc_bounds)(lambda x: oscillator(t, x[0], x[1]))[0] for t in ts]
    )
    fs_t_star = np.array(
        [oscillator(t, 0.5 * (x1_lp + x1_up), 0.5 * (x2_lp + x2_up)) for t in ts]
    )
    lps: Sequence[NaturalExtensionResult] = [
        result.semi[error][-1] for result in simulation.values()
    ]
    plt.plot(ts, fs_t_star, color="C1", linestyle="--", label=r"$f_\tau(t^*)$")
    plt.plot(
        ts,
        [x.alpha for x in lps],
        color="C0",
        linestyle="-",
        label=r"$E̲(f_\tau(X_1,X_2)$",
    )
    plt.plot(ts, inf_fs, color="C2", linestyle=":", label=r"$\inf_{t\in T} f_\tau(t)$")
    plt.legend()
    plt.xlabel(r"$\tau$")
    plt.tight_layout()
    plt.savefig("plot-alpha-bound-1.png")
    plt.close()
    # lambda
    plt.plot(
        ts,
        fs_t_star - inf_fs,
        color="C1",
        label=r"$f_\tau(t^*)-\inf_{t\in T} f_\tau(t)$",
        linestyle="--",
    )
    # plot code assumes it is 0.1 for simplicity...
    assert x1_up - x1_lp == pytest.approx(0.1)
    assert x2_up - x2_lp == pytest.approx(0.1)
    assert all(len(x.lambda_) == 4 for x in lps)
    plt.plot(
        ts,
        [sum(x.lambda_) * 0.05 for x in lps],
        color="C0",
        linestyle="-",
        label=r"$\sum_X\lambda_X(X(t^*)-P̲(X))$",
    )
    plt.hlines(0, min(ts), max(ts), color="C2", linestyle=":", label="0")
    plt.legend()
    plt.xlabel(r"$\tau$")
    plt.tight_layout()
    plt.savefig("plot-alpha-bound-2.png")
    plt.close()


def plot_time_delta_iters(simulation: Mapping[float, SimulationResult]) -> None:
    ts: Sequence[float] = list(simulation.keys())
    nums: Sequence[int] = list(simulation[ts[0]].grid.keys())
    errors: Sequence[float] = list(simulation[ts[0]].semi.keys())
    line_styles = ["-", "--", ":"]
    for error, line_style in zip(errors, line_styles):
        plt.plot(
            ts,
            [
                sum(res.time for res in result.semi[error])
                for result in simulation.values()
            ],
            color="C0",
            linestyle=line_style,
            label=rf"$\epsilon_1=\epsilon_2={error:g}$",
        )
    for num, line_style in zip(nums, line_styles):
        plt.plot(
            ts,
            [result.grid[num].time for result in simulation.values()],
            color="C1",
            linestyle=line_style,
            label=rf"$|U|={num ** 2}$",
        )
    plt.legend()
    plt.yscale("log")
    plt.xlabel(r"$\tau$")
    plt.ylabel("computing time")
    plt.tight_layout()
    plt.savefig("plot-time.png")
    plt.close()

    for error, line_style in zip(errors, line_styles):
        plt.plot(
            ts,
            [result.semi[error][-1].delta_tilde for result in simulation.values()],
            color="C0",
            linestyle=line_style,
            label=rf"$\epsilon_1=\epsilon_2={error:g}$",
        )
    for num, line_style in zip(nums, line_styles):
        plt.plot(
            ts,
            [result.grid[num].delta_tilde for result in simulation.values()],
            color="C1",
            linestyle=line_style,
            label=rf"$|U|={num ** 2}$",
        )
    plt.legend()
    plt.yscale("log")
    plt.ylim(bottom=1e-7)
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\tilde{\delta}$")
    plt.tight_layout()
    plt.savefig("plot-delta-tilde.png")
    plt.close()

    for error, line_style in zip(errors, line_styles):
        plt.plot(
            ts,
            [
                len(result.semi[error])  # |U_k| is number of iterations
                for result in simulation.values()
            ],
            color="C0",
            linestyle=line_style,
            label=rf"$\epsilon_1=\epsilon_2={error:g}$",
        )
    plt.legend()
    plt.xlabel(r"$\tau$")
    plt.ylabel("$|U_k|$")
    plt.tight_layout()
    plt.savefig("plot-iterations.png")
    plt.close()


def plot_points(
    simulation2: Mapping[float, Sequence[NaturalExtensionResult]],
    error: float,
    tag: str,
) -> None:
    ts = list(simulation2.keys())
    results = list(simulation2.values())
    logging.info("plotting points for t in %s", ts)
    plt.fill(
        [x1_lb, x1_lb, x1_ub, x1_ub],
        [x2_lb, x2_ub, x2_ub, x2_lb],
        color="black",
        alpha=0.1,
    )
    colors = ["C0", "C1", "C2"]
    markers = ["o", "s", "D"]
    line_styles = ["-", "--", ":"]
    plt.scatter(
        [0.5 * (x1_lp + x1_up)],
        [0.5 * (x2_lp + x2_up)],
        color="black",
        marker="+",
        label=r"$t^*$",
    )
    for t, result, color, marker in zip(ts, results, colors, markers):
        points = [res.t_next for res in result]
        plt.scatter(
            [p[0] for p in points],
            [p[1] for p in points],
            color=color,
            marker=marker,
            label=rf"$U_k$ when $\tau={t:g}$",
        )
    plt.legend()
    plt.xlabel("$t_1$")
    plt.ylabel("$t_2$")
    plt.tight_layout()
    plt.savefig(f"plot-points-{tag}.png")
    plt.close()

    for t, result, color, line_style in zip(ts, results, colors, line_styles):
        deltas = [res.delta_tilde for res in result]
        plt.plot(
            range(len(deltas)),
            deltas,
            color=color,
            linestyle=line_style,
            label=rf"when $\tau={t:g}$",
        )
    plt.axhline(y=error, color="C3", linestyle="-.", label=f"{error:g}")
    plt.legend()
    plt.xlabel("$k$")
    plt.ylabel(r"$\tilde{\delta}$")
    plt.yscale("log")
    plt.ylim(bottom=0.5 * 1e-6)
    plt.tight_layout()
    plt.savefig(f"plot-delta-tilde-iter-{tag}.png")
    plt.close()


def get_osc_lin_prog(t: float, num: int, min_fun: MinFun) -> NaturalExtensionResult:
    logging.info("get_osc_lin_prog %s %s", t, num)
    grid = np.meshgrid(
        *[np.linspace(lb, ub, num) for lb, ub in zip(osc_bounds.lb, osc_bounds.ub)]
    )
    points = list(np.vstack([coord.ravel() for coord in grid]).T)
    y: Gamble = osc_y(t)
    result = solve_natural_extension_1(
        y, osc_low_prev, points, min_fun_brute(osc_bounds)
    )
    return result


def get_osc_semi_lin_prog(
    t: float, error: float, min_fun: Callable[[Sequence[npt.NDArray]], MinFun]
) -> Sequence[NaturalExtensionResult]:
    logging.info("get_osc_semi_lin_prog %s %s", t, error)
    points = [np.array([0.5 * (x1_lp + x1_up), 0.5 * (x2_lp + x2_up)])]
    y: Gamble = osc_y(t)
    return list(solve_natural_extension_2(y, osc_low_prev, points, min_fun, error))


def osc_min_fun(points: Sequence[npt.NDArray], min_fun: MinFun) -> MinFun:
    def _(fun: Callable[[npt.NDArray], float]) -> tuple[float, npt.NDArray]:
        return min(
            [min_fun_minimize(osc_bounds, x0)(fun) for x0 in points] + [min_fun(fun)]
        )

    return _


def load_simulation(slow=True) -> Mapping[float, SimulationResult]:
    simulation_file = Path("simulation.pickle")
    if simulation_file.exists():
        with simulation_file.open("rb") as rfile:
            return pickle.load(rfile)
    nums = [10, 40, 160] if slow else [10]
    errors = [1e-2, 1e-4, 1e-6] if slow else [1e-6]
    min_grid = min_fun_brute(osc_bounds)
    min_semi: Callable[[Sequence[npt.NDArray]], MinFun] = functools.partial(
        osc_min_fun, min_fun=min_grid
    )
    simulation: Mapping[float, SimulationResult] = {
        t: SimulationResult(
            grid={num: get_osc_lin_prog(t, num, min_grid) for num in nums},
            semi={error: get_osc_semi_lin_prog(t, error, min_semi) for error in errors},
        )
        for t in np.linspace(0, 2, 201 if slow else 11)
    }
    with simulation_file.open("wb") as out_file:
        pickle.dump(simulation, out_file)
    return simulation


def load_simulation_2() -> Mapping[float, Sequence[NaturalExtensionResult]]:
    simulation_file = Path("simulation2.pickle")
    if simulation_file.exists():
        with simulation_file.open("rb") as rfile:
            return pickle.load(rfile)
    min_grid = min_fun_brute(osc_bounds, 1000)
    min_semi: Callable[[Sequence[npt.NDArray]], MinFun] = functools.partial(
        osc_min_fun, min_fun=min_grid
    )
    simulation: Mapping[float, Sequence[NaturalExtensionResult]] = {
        t: get_osc_semi_lin_prog(t, 1e-6, min_semi) for t in [0.25, 0.5, 0.75]
    }
    with simulation_file.open("wb") as out_file:
        pickle.dump(simulation, out_file)
    return simulation


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _simulation = load_simulation()
    plot_alpha_bound(simulation=_simulation, error=1e-6)
    plot_time_delta_iters(simulation=_simulation)
    _simulation2 = {
        t: result.semi[1e-6]
        for t, result in _simulation.items()
        if t in {0.25, 0.5, 0.75}
    }
    assert list(_simulation2.keys()) == [0.25, 0.5, 0.75]
    plot_points(simulation2=_simulation2, error=1e-6, tag="1")
    _simulation2_alt = load_simulation_2()
    plot_points(simulation2=_simulation2_alt, error=1e-6, tag="2")
    # plot_oscillator(t=0, num=30, cmap="plasma")
    # plot_oscillator(t=1, num=30, cmap="plasma")
    # plot_for_modulus(t=2, zs=np.linspace(0, 0.1, 10))
    # plot_for_modulus(t=5, zs=np.linspace(0, 0.1, 30))
