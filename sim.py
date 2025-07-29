import functools
import logging
import math
import pickle
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
from matplotlib import colors
from scipy.optimize import Bounds

from lowprev_lsip.low_prev import (
    Gamble,
    GambleGrad,
    NaturalExtensionResult,
    discrepancy,
    get_conjugate_gamble,
    solve_natural_extension_1,
    solve_natural_extension_2,
)
from lowprev_lsip.modulus import (
    lipschitz_constant,
    modulus_of_continuity,
)
from lowprev_lsip.optimize import (
    MinFun,
    min_fun_brute,
    min_fun_differential_evolution,
    min_fun_direct,
    min_fun_dual_annealing,
    min_fun_shgo,
)

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


def plot_oscillator(t: float, num: int, cmap: str, tag: str) -> None:
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
    plt.tight_layout()
    plt.savefig(f"plot-oscillator-{tag}.png")
    plt.close()


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
    plt.grid()
    plt.tight_layout()
    plt.savefig("plot-bound-alpha.png")
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
        label=r"$0.05\sum_X\lambda_X(X(t^*)-P̲(X))$",
    )
    plt.hlines(0, min(ts), max(ts), color="C2", linestyle=":", label="0")
    plt.legend()
    plt.xlabel(r"$\tau$")
    plt.grid()
    plt.tight_layout()
    plt.savefig("plot-bound-lambda.png")
    plt.close()


def plot_time_delta_iters(
    simulation: Mapping[float, SimulationResult], tag: str
) -> None:
    ts: Sequence[float] = list(simulation.keys())
    nums: Sequence[int] = list(simulation[ts[0]].grid.keys())
    errors: Sequence[float] = list(simulation[ts[0]].semi.keys())
    line_styles = ["-", "--", ":"]
    # note: omit first time point to remove "warmup" outlier in timing
    for error, line_style in zip(errors, line_styles):
        plt.plot(
            ts[1:],
            [
                sum(res.time for res in result.semi[error])
                for result in simulation.values()
            ][1:],
            color="C0",
            linestyle=line_style,
            label=rf"$\epsilon_1=\epsilon_2={error:g}$",
        )
    for num, line_style in zip(nums, line_styles):
        plt.plot(
            ts[1:],
            [result.grid[num].time for result in simulation.values()][1:],
            color="C1",
            linestyle=line_style,
            label=rf"$|U|={num ** 2}$",
        )
    plt.legend()
    plt.yscale("log")
    plt.xlabel(r"$\tau$")
    plt.ylabel("computing time")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plot-sim-time-{tag}.png")
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
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plot-sim-delta-tilde-{tag}.png")
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
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plot-sim-iterations-{tag}.png")
    plt.close()


def plot_points(
    simulation: Mapping[float, SimulationResult],
    error: float,
    times: Collection[float],
    tag: str,
) -> None:
    simulation2 = {
        t: result.semi[error] for t, result in simulation.items() if t in times
    }
    assert list(simulation2.keys()) == list(times)
    ts = list(simulation2.keys())
    results = list(simulation2.values())
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
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plot-sim-points-{tag}.png")
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
    plt.legend()
    plt.xlabel("$k$")
    plt.ylabel(r"$\tilde{\delta}_k$")
    plt.yscale("log")
    plt.ylim(bottom=0.5 * 1e-6)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plot-sim-delta-tilde-iter-{tag}.png")
    plt.close()


def plot_discrepancy(
    result: NaturalExtensionResult,
    t: float,
    k: int | None,
    num: int,
    cmap: str,
    tag: str,
) -> None:
    x = np.linspace(osc_bounds.lb[0], osc_bounds.ub[0], num)
    y = np.linspace(osc_bounds.lb[1], osc_bounds.ub[1], num)
    xx, yy = np.meshgrid(x, y)
    fun = np.frompyfunc(
        lambda x1, x2: discrepancy(
            osc_y(t), osc_low_prev, result.lambda_, result.alpha, np.array([x1, x2])
        ),
        2,
        1,
    )
    zz = fun(xx, yy).astype(float)
    plt.contourf(xx, yy, zz, cmap=cmap, norm=colors.TwoSlopeNorm(vcenter=0))
    plt.colorbar(label=r"$H_{{\lambda,\alpha}}(t_1,t_2)$")
    plt.xlabel("$t_1$")
    plt.ylabel("$t_2$")
    plt.title(rf"$\tau={t}$" + (f" and $k={k}$" if k is not None else ""))
    plt.tight_layout()
    plt.savefig(f"plot-sim-h-{tag}.png")
    plt.close()


def get_osc_lin_prog(t: float, num: int, min_fun: MinFun) -> NaturalExtensionResult:
    logging.info("get_osc_lin_prog %s %s", t, num)
    grid = np.meshgrid(
        *[np.linspace(lb, ub, num) for lb, ub in zip(osc_bounds.lb, osc_bounds.ub)]
    )
    points = list(np.vstack([coord.ravel() for coord in grid]).T)
    y: Gamble = osc_y(t)
    result = solve_natural_extension_1(y, osc_low_prev, points, min_fun)
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
        return min_fun(fun)

    return _


def load_simulation(
    min_grid: MinFun, tag: str, slow=True
) -> Mapping[float, SimulationResult]:
    simulation_file = Path(f"simulation-{tag}.pickle")
    if simulation_file.exists():
        with simulation_file.open("rb") as rfile:
            return pickle.load(rfile)
    nums = [10, 50, 250] if slow else [10]
    errors = [1e-2, 1e-4, 1e-6] if slow else [1e-6]
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("plotting function")
    plot_oscillator(t=0, num=300, cmap="plasma", tag="0_0")
    plot_oscillator(t=0.5, num=300, cmap="plasma", tag="0_5")
    plot_oscillator(t=1, num=300, cmap="plasma", tag="1_0")
    plot_oscillator(t=2, num=300, cmap="plasma", tag="2_0")
    _simulations: dict[str, Mapping[float, SimulationResult]] = {}
    for _tag, _min_grid in [
        ("brute3", min_fun_brute(osc_bounds, ns=3)),
        ("brute20", min_fun_brute(osc_bounds, ns=20)),
        ("brute100", min_fun_brute(osc_bounds, ns=100)),
        ("evol", min_fun_differential_evolution(osc_bounds)),
        ("shgo", min_fun_shgo(osc_bounds)),
        ("anneal", min_fun_dual_annealing(osc_bounds)),
        ("direct", min_fun_direct(osc_bounds)),
    ]:
        logging.info("simulating %s", _tag)
        _simulation = load_simulation(min_grid=_min_grid, tag=_tag)
        _simulations[_tag] = _simulation
    for _tag, _sim in _simulations.items():
        logging.info("plotting %s", _tag)
        plot_time_delta_iters(simulation=_simulation, tag=_tag)
        plot_points(
            simulation=_simulation, error=1e-6, times={0.25, 0.5, 0.75}, tag=_tag
        )
        _t = 0.25
        _error = 1e-6
        for _k, _result in enumerate(_sim[_t].semi[_error]):
            plot_discrepancy(
                result=_result,
                t=_t,
                k=_k,
                num=300,
                cmap="coolwarm",
                tag=f"{_tag}-semi-{_k}",
            )
        for _num, _result in _sim[_t].grid.items():
            plot_discrepancy(
                result=_result,
                t=_t,
                k=None,
                num=300,
                cmap="coolwarm",
                tag=f"{_tag}-grid-{_num}",
            )
    plot_alpha_bound(simulation=_simulations["brute100"], error=1e-6)
    # plot_for_modulus(t=2, zs=np.linspace(0, 0.1, 10))
    # plot_for_modulus(t=5, zs=np.linspace(0, 0.1, 30))
