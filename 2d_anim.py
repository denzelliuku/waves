"""
Animation of a two dimensional (with regards to the spatial dimensions) wave(s)
that have a sinusoidal form as shown in Wikipedia:
https://en.wikipedia.org/wiki/Sinusoidal_plane_wave
"""

import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from time import perf_counter
from typing import Callable, Any
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


def wavefun(x: np.ndarray, t: int | float | np.ndarray, n: np.ndarray,
            amp: int | float, k: int | float, w: int | float,
            delta: int | float) -> np.ndarray:
    """
    :param x:
    :param t:
    :param n:
    :param amp:
    :param k:
    :param w:
    :param delta:
    :return:
    """
    return amp * np.cos(k * np.dot(x, n) - w * t + delta)


def _update_anim(n: int, conts: list[plt.contourf], ax: plt.axes, x: np.ndarray,
                 y: np.ndarray, data: np.ndarray) -> plt.contourf:
    """
    Updates the plots
    :param n:
    :param conts:
    :param x:
    :param y:
    :param data:
    :return:
    """
    for c in conts[0].collections:
        c.remove()
    conts[0] = ax.contourf(x, y, data[n, :, :])
    return conts[0].collections


@timer
def animate(funcs: partial,  x: np.ndarray, y: np.ndarray, t: np.ndarray) -> None:
    """
    :param funcs:
    :param x:
    :param y:
    :param t:
    :return:
    """
    coords = np.zeros(shape=(x.shape[0], y.shape[0], 2))
    for j, yp in enumerate(y):
        for i, xp in enumerate(x):
            coords[j, i, 0] = xp
            coords[j, i, 1] = yp
    data = np.zeros(shape=(t.shape[0], x.shape[0], y.shape[0]))
    for i, tp in enumerate(t):
        data[i, :, :] = funcs(coords, tp)
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(xlim=(min(x), max(x)), ylim=(min(y), max(y)), xlabel='x',
                  ylabel='y')
    conts = [ax.contourf(x, y, data[0])]
    anim = FuncAnimation(fig=fig, func=_update_anim, frames=len(t),
                         fargs=(conts, ax, x, y, data), blit=True)
    anim.save(filename='2danim.gif', writer='pillow', fps=60, dpi=100)


def main() -> None:
    f = 20  # Frequency [1/s]
    w = 2 * np.pi * f  # Angular frequency [1/s]
    wavelength = .1  # [m]
    k = 2 * np.pi / wavelength  # Wave number
    amp = 2  # Amplitude [m]
    delta = 0  # Phase shift [rad]
    x = y = np.arange(0, 1, .001)  # Spatial coordinates (grid)
    t = np.arange(0, 1, .01)  # Temporal coordinates (timesteps
    n = np.array([.25, .5])  # Direction of propagation
    wave = partial(wavefun, n=n, amp=amp, k=k, w=w, delta=delta)
    animate(wave, x=x, y=y, t=t)


if __name__ == '__main__':
    main()
