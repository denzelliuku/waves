"""
Animation of one dimensional (with regards to space) waves that have a wave
function of form f(x, t) = A*cos(k*x - w*t + delta)
"""
import functools

import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from time import perf_counter
from typing import Any, Callable
from matplotlib.animation import FuncAnimation


def timer(f: Callable) -> Any:
    """
    Decorator to time a function
    :param f:
    :return:
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        s = perf_counter()
        rv = f(*args, **kwargs)
        print(f"Function '{f.__name__}' ran in {perf_counter() - s:.3f} s")
        return rv
    return wrapper


def wavefun(x: np.ndarray, t: np.ndarray, amp: int | float, k: int | float,
            w: int | float, delta: int | float) -> np.ndarray:
    """
    :param x:
    :param t:
    :param amp:
    :param k:
    :param w:
    :param delta:
    :return:
    """
    return amp * np.cos(k * x - w * t + delta)


def _init_anim(lines: list[plt.Line2D]) -> list[plt.Line2D]:
    """
    Initialises the plots (basically setting the coordinates to empty)
    :param lines:
    :return:
    """
    for i, line in enumerate(lines):
        line.set_data([], [])
    return lines


def _update_anim(i: int, lines: list[plt.Line2D], x: np.ndarray,
                 data: np.ndarray) -> list[plt.Line2D]:
    """
    Updates the plots
    :param i:
    :param lines:
    :param x:
    :param data:
    :return:
    """
    left_y = data[0, i, :]
    right_y = data[1, i, :]
    summa_y = data[2, i, :]
    lines[0].set_data(x, left_y)
    lines[1].set_data(x, right_y)
    lines[2].set_data(x, summa_y)
    return lines


@timer
def animate(*funcs: functools.partial,  x: np.ndarray, t: np.ndarray) -> None:
    """
    :param funcs:
    :param x:
    :param t:
    :return:
    """
    xx, tt = np.meshgrid(x, t)
    n = len(funcs)
    data = np.zeros(shape=(n + 1, *xx.shape))
    for i, fun in enumerate(funcs):
        data[i, :, :] = fun(xx, tt)
    data[-1, :, :] = np.sum(data, axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid()
    colors = ['b', 'g', 'r', 'y']
    alpha = .75
    # Lines for the individual waves
    lines = [ax.plot([], [], c=colors[i % n], ls='--', alpha=alpha)[0]
             for i in range(n)]
    # Line for the sum wave
    lines.append(ax.plot([], [], c='k')[0])
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(data[-1]), np.max(data[-1]))
    # lines = _init_anim(lines)
    anim = FuncAnimation(fig=fig, func=_update_anim, frames=len(t),
                         fargs=(lines, x, data), blit=True)
    anim.save(filename='yeet.gif', writer='pillow', fps=60, dpi=100)


def main():
    f = 5  # [1/s]
    w = 2 * np.pi * f
    wavelength = 2  # [m]
    k = 2 * np.pi / wavelength
    amp = 2
    delta = 0
    x = np.arange(0, 2 * np.pi, .01)
    t = np.arange(0, 2 * np.pi, .005)
    wave1 = partial(wavefun, amp=amp, k=k, w=w, delta=delta)
    wave2 = partial(wavefun, amp=amp, k=k, w=-w, delta=delta)
    animate(wave1, wave2, x=x, t=t)


if __name__ == '__main__':
    main()
