"""
Animation of one dimensional (with regards to the spatial dimensions) waves
that have a wave function of form f(x, t) = A*cos(k*x - w*t + delta)
"""

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


def wavefun(x: int | float | np.ndarray, t: int | float | np.ndarray,
            amp: int | float, k: int | float, w: int | float,
            delta: int | float) -> np.ndarray:
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


def gaussian(x: int | float | np.ndarray, t: int | float | np.ndarray,
             wlength: int | float, c: int | float) -> np.ndarray:
    """
    The real part of a wave packet taken straight from wikipedia:
    https://en.wikipedia.org/wiki/Wave_packet
    :param x:
    :param t:
    :param wlength:
    :param c:
    :return:
    """
    xx = 2 * np.pi * (x - c * t) / wlength
    return np.exp(-np.power(x - c * t, 2)) * np.cos(xx)


def _update_anim(n: int, lines: list[plt.Line2D], x: np.ndarray,
                 data: np.ndarray) -> list[plt.Line2D]:
    """
    Updates the plots
    :param n:
    :param lines:
    :param x:
    :param data:
    :return:
    """
    for i, line in enumerate(lines):
        y = data[i, n, :]
        line.set_data(x, y)
    plt.title(f"Time: {n}")
    return lines


@timer
def animate(*funcs: partial,  x: np.ndarray, t: np.ndarray) -> None:
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
    colors = ["b", "g", "r", "y", "m"]
    alpha = .5
    # Lines for the individual waves
    lines = [ax.plot([], [], c=colors[i % len(colors)], ls="--", alpha=alpha)[0]
             for i in range(n)]
    # Line for the sum wave separately with different alpha and style
    lines.append(ax.plot([], [], c="k")[0])
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(data[-1]), np.max(data[-1]))
    anim = FuncAnimation(fig=fig, func=_update_anim, frames=len(t),
                         fargs=(lines, x, data), blit=True)
    anim.save(filename="1danim.gif", writer="pillow", fps=60, dpi=100)


def main() -> None:
    f = 10  # Frequency [1/s]
    w = 2 * np.pi * f  # Angular frequency [1/s]
    wavelength = .5  # [m]
    k = 2 * np.pi / wavelength  # Wave number
    amp = 2  # Amplitude [m]
    delta = 0  # Phase shift [rad]
    x = np.arange(0, 2 * np.pi, .01)  # Spatial coordinates
    t = np.arange(0, 2 * np.pi, .005)  # Temporal coordinates (timesteps)
    # waves = []
    # for i in range(8):
    #     n = 2 ** i
    #     sign = 1 if i % 2 == 0 else -1
    #     wave = partial(wavefun, amp=amp/n, k=k*n, w=n*w*sign, delta=delta)
    #     waves.append(wave)
    wave = partial(gaussian, wlength=wavelength, c=(w / k))
    animate(wave, x=x, t=t)


if __name__ == "__main__":
    main()
