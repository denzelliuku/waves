"""
Animation of a two dimensional (with regards to the spatial dimensions) wave(s)
that have a sinusoidal form as shown in Wikipedia:
https://en.wikipedia.org/wiki/Sinusoidal_plane_wave
"""
import matplotlib.image
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


def _update_anim(n: int, im: matplotlib.image.AxesImage, ax: plt.axes,
                 data: np.ndarray) -> plt.contourf:
    """
    Updates the plots
    :param n:
    :param im:
    :param ax:
    :param data:
    :return:
    """
    im.set_array(data[n])
    ax.set_title(f"Frame: {n}")
    return im,


@timer
def animate(*funcs: partial,  x: np.ndarray, y: np.ndarray, t: np.ndarray) -> None:
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
        for fun in funcs:
            data[i, :, :] += fun(coords, tp)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    im = plt.imshow(data[0], cmap="winter", vmin=np.min(data),
                    vmax=np.max(data))
    ax.axis("off")
    anim = FuncAnimation(fig=fig, func=_update_anim, frames=len(t),
                         fargs=(im, ax, data), blit=True)
    anim.save(filename="2danim.gif", writer="pillow", fps=60)


def _norm(v: np.ndarray) -> np.ndarray:
    """
    Normalises the given vector
    :param v:
    :return:
    """
    return v / np.linalg.norm(v)


def main() -> None:
    f = 20  # Frequency [1/s]
    w = 2 * np.pi * f  # Angular frequency [1/s]
    wavelength = 1  # [m]
    k = 2 * np.pi / wavelength  # Wave number
    amp = 2  # Amplitude [m]
    delta = 0  # Phase shift [rad]
    x = y = np.arange(0, 5, .005)  # Spatial coordinates (the grid)
    t = np.arange(0, .5, .005)  # Temporal coordinates (timesteps)
    n1 = np.array([0, 1])  # Direction of propagation(s)
    n2 = np.array([0, 1])
    wave1 = partial(wavefun, n=_norm(v=n1), amp=amp, k=k, w=w, delta=delta)
    wave2 = partial(wavefun, n=_norm(v=n2), amp=amp, k=k, w=w, delta=delta)
    animate(wave1, wave2, x=x, y=y, t=t)
    # xx, yy = np.meshgrid(x, y)
    # z1 = np.sin(2*np.pi*1*np.sqrt((xx - 0.5) * (xx - 0.5) + yy * yy))
    # z2 = np.sin(2*np.pi*1*np.sqrt((xx + 0.5) * (xx + 0.5) + yy * yy))
    # z = z1 + z2
    # plt.imshow(z)
    # plt.show()


if __name__ == "__main__":
    main()
