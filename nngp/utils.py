import numpy as np


def piecewise1(num_points: int = 25, random_seed: int = None) -> np.ndarray:

    def f(x):
        return np.piecewise(
            x, [x < 1.7, x >= 1.7],
            [lambda x: x**2.5, lambda x: 0.9*x**1.2])
    
    start = 0
    stop = 3

    if random_seed is not None:
        np.random.seed(random_seed)
        x = np.random.uniform(start, stop, num_points)
    else:
        x = np.linspace(start, stop, num_points)
    x_test = np.linspace(start, stop, 200)

    return x, x_test, f(x), f(x_test)


def piecewise2(num_points: int = 25, random_seed: int = None):

    start = 0
    stop = 10
    
    def f(x):
        return np.piecewise(
            x, [x < 5, x >= 5],
            [lambda x: np.sin(x), lambda x: np.sin(x) + 3])

    if random_seed is not None:
        np.random.seed(random_seed)
        x = np.random.uniform(start, stop, num_points)
    else:
        x = np.linspace(start, stop, num_points)
    x_test = np.linspace(start, stop, 200)

    return x, x_test, f(x), f(x_test)


def piecewise3(num_points: int = 25, random_seed: int = None) -> np.ndarray:

    def f(x):
        return np.piecewise(
            x, [x < 1, (x >= 1) & (x < 2), x >= 2],
            [lambda x: 1 * x**2.5,
             lambda x: 0.5 * x**1.5 - 2,
             lambda x: 1 * np.exp(0.5 * (x - 2)) + 1]
             )

    start = 0
    stop = 3

    if random_seed is not None:
        np.random.seed(random_seed)
        x = np.random.uniform(start, stop, num_points)
    else:
        x = np.linspace(start, stop, num_points)
    x_test = np.linspace(start, stop, 200)

    return x, x_test, f(x), f(x_test)


def nonstationary1(num_points: int = 25, random_seed: int = None):

    start = 0.2
    stop = 4.8
    
    def f(x):
        transition = 1 / (1 + np.exp(-10 * (x - np.pi))) 
        smooth_part = np.sin(2 * x)
        non_smooth_part = 0.3 * np.sin(30 * x) + 0.3 * np.cos(10 * x)
        return (1 - transition) * smooth_part + transition * non_smooth_part

    if random_seed is not None:
        np.random.seed(random_seed)
        x = np.random.uniform(start, stop, num_points)
    else:
        x = np.linspace(start, stop, num_points)
    x_test = np.linspace(start, stop, 200)

    return x, x_test, f(x), f(x_test)


def nonstationary2(num_points: int = 25, random_seed: int = None) -> np.ndarray:

    start = -7
    stop = 7
    
    def f(x):
        y_smooth = np.sin(0.7 * x) * (np.abs(x) >= 2)
        y_non_smooth = np.sin(10 * x) * np.exp(-np.abs(2 * x)) * (np.abs(x) < 2)
        return y_smooth + y_non_smooth

    if random_seed is not None:
        np.random.seed(random_seed)
        x = np.random.uniform(start, stop, num_points)
    else:
        x = np.linspace(start, stop, num_points)
    x_test = np.linspace(start, stop, 200)

    return x, x_test, f(x), f(x_test)


def nonstationary3(num_points: int = 25, random_seed: int = None):

    start = 0
    stop = 10
    
    def f(x):
        spike_params = [
            (1, 2, 0.1),   # (amplitude, position, width)
            (-0.5, 4, 0.15),
            (1.5, 6, 0.1),
            (-1, 8, 0.1),
            (0.75, 9, 0.1)
        ]
        # Generate the smooth base curve
        y = np.sin(x)
        # Add each spike to the base curve
        for a, b, c in spike_params:
            y += a * np.exp(-((x - b)**2) / (2 * c**2))
        return y

    if random_seed is not None:
        np.random.seed(random_seed)
        x = np.random.uniform(start, stop, num_points)
    else:
        x = np.linspace(start, stop, num_points)
    x_test = np.linspace(start, stop, 200)

    return x, x_test, f(x), f(x_test)
