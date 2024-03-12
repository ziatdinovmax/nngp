import numpy as np


def piecewise1(x: np.ndarray, params) -> np.ndarray:
    return np.piecewise(
        x,
        [x < params["t"], x >= params["t"]],
        [lambda x: x**params["beta1"], lambda x: 0.5*x**params["beta2"]])


def piecewise2(x: np.ndarray):
    return np.piecewise(
        x, [x < 5, x >= 5],
        [lambda x: np.sin(x), lambda x: np.sin(x) + 3])


def piecewise3(x: np.ndarray) -> np.ndarray:
    return np.piecewise(
        x,
        [x < 1, (x >= 1) & (x < 2), x >= 2],
        [
            lambda x: 1 * x**2.5,
            lambda x: 0.5 * x**1.5 - 2,
            lambda x: 1 * np.exp(0.5 * (x - 2)) + 1
        ]
    )


def nonsmooth1(x: np.ndarray) -> np.ndarray:
    y_smooth = np.sin(0.7 * x) * (np.abs(x) >= 2)
    y_non_smooth = np.sin(10 * x) * np.exp(-np.abs(2 * x)) * (np.abs(x) < 2)
    return y_smooth + y_non_smooth


def nonsmooth2(x):
    transition = 1 / (1 + np.exp(-10 * (x - np.pi)))    
    smooth_part = np.sin(2 * x)    
    non_smooth_part = 0.1 * np.sin(10 * x) + 0.1 * np.cos(20 * x) + 0.3 * np.sin(x)**2    
    return (1 - transition) * smooth_part + transition * non_smooth_part


