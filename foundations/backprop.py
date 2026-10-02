import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        
        logits = x @ w + b
        sigmoid = lambda z: 1 / (1 + np.exp(-z))
        y_hat = sigmoid(logits)
        loss = 0.5 * np.power((y_hat - y_true), 2)
        
        dl_dy = y_hat - y_true
        dy_dz = (y_hat) * (1 - y_hat)
        dz_dw = x
        dz_db = 1

        dl_dw = dl_dy * dy_dz * dz_dw
        dl_db = dl_dy * dy_dz * dz_db

        return (np.round(dl_dw, 5), np.round(dl_db, 5))

