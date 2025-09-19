import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.preprocessing import sg_first_derivative, sg_second_derivative


def test_savgol_derivatives_do_not_raise():
    X = np.tile(np.linspace(0.0, 1.0, 21, dtype=float), (3, 1))

    first = sg_first_derivative(X)
    second = sg_second_derivative(X)

    assert first.shape == X.shape
    assert second.shape == X.shape
