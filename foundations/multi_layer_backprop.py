import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        # list to np.arr
        x = np.array(x)
        W1, b1 = np.array(W1), np.array(b1)
        W2, b2 = np.array(W2), np.array(b2)
        y_true = np.array(y_true)

        # forward pass
        l1 = W1 @ x + b1
        act = np.maximum(0, l1)
        y_hat = W2 @ act + b2

        # loss calculation
        loss = np.mean(np.power(y_hat - y_true, 2))

        # gradient calculation
        
        # xw1+b1 -> act -> actw2+b2 -> loss

        # loss -> actw2+b2 -> act -> xw1+b1 

        # dl_dw2 = dl/dy_hat * dy_hat/dw2 
        n = len(y_true)
        dl_dy_hat = 2 / n * (y_hat - y_true)
        dy_hat_dw2 = act
        dy_hat_db2 = 1

        dl_dw2 = np.outer(dl_dy_hat, dy_hat_dw2)
        dl_db2 = dl_dy_hat * dy_hat_db2

        # dl_dw1 = dl/dy_hat * dy_hat/dact * dact/dl1 * dl1/dw1
        dy_hat_dact = dl_dy_hat @ W2
        d_act_dl1 = dy_hat_dact * (l1 > 0)
        dl1_dw1 = x
        dl1_db1 = 1

        dl_dw1 = np.outer(d_act_dl1 ,dl1_dw1)
        dl_db1 = d_act_dl1 * dl1_db1

        res = {
        'loss':  np.round(loss, 4),
        'dW1':   np.round(dl_dw1, 4),
        'db1':   np.round(dl_db1,4),
        'dW2':   np.round(dl_dw2, 4),
        'db2':   np.round(dl_db2, 4)
        }

        return res









