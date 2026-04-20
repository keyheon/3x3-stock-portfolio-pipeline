"""
models.py — Matrix Network + Differentiable Sinkhorn

Used for Step 4 (3x3 allocation matrix).
Other models (SelfAttention, StockScorerNN, StockScorerEnsemble) were removed
in the release cleanup because their outputs were not consumed by the pipeline.
"""

import numpy as np
from config import *


def sinkhorn_3x3(logits, row_m, col_m, n_iter=SINKHORN_ITERS):
    """
    Differentiable Sinkhorn normalization.
    Converts 9-dim logits into a 3x3 matrix with row sums = row_m, col sums = col_m.
    All ops are differentiable so gradients can flow through this layer.
    """
    K = np.exp(logits.reshape(3, 3) - logits.max())
    K = np.maximum(K, 1e-12)
    for _ in range(n_iter):
        K *= (np.array(row_m).reshape(-1, 1) / np.maximum(K.sum(1, keepdims=True), 1e-12))
        K *= (np.array(col_m).reshape(1, -1) / np.maximum(K.sum(0, keepdims=True), 1e-12))
    return K


class MatrixNetwork:
    """
    3x3 Matrix Network (End-to-End).
    Architecture: Input -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(9) -> Sinkhorn -> 3x3 matrix.
    Input = selected_stocks * features (flattened).
    Output = 3x3 allocation matrix that respects marginals.
    Gradients propagate: portfolio loss -> Sinkhorn -> hidden layers.
    """

    def __init__(self, input_dim, seed=RANDOM_SEED):
        np.random.seed(seed)
        sizes = [input_dim, MATRIX_HIDDEN_1, MATRIX_HIDDEN_2, MATRIX_OUTPUT]

        self.layers = []
        for i in range(len(sizes) - 1):
            fi, fo = sizes[i], sizes[i+1]
            self.layers.append({
                'W': np.random.randn(fi, fo) * np.sqrt(2 / fi),
                'b': np.zeros((1, fo)),
                'mW': None, 'vW': None, 'mb': None, 'vb': None,  # Adam states
            })

        self.t_adam = 0
        self.architecture = ' -> '.join(str(s) for s in sizes)

    def forward(self, X, row_m, col_m):
        """Forward pass: features -> hidden layers -> logits -> Sinkhorn -> 3x3 matrix."""
        self.cache = {'activations': [X.reshape(1, -1)]}
        current = self.cache['activations'][0]

        for i, layer in enumerate(self.layers):
            z = current @ layer['W'] + layer['b']
            if i < len(self.layers) - 1:
                current = np.maximum(0, z)  # ReLU
            else:
                current = z  # linear (raw logits)
            self.cache['activations'].append(current)

        self.cache['logits'] = current.flatten()
        matrix = sinkhorn_3x3(self.cache['logits'], row_m, col_m)
        self.cache['matrix'] = matrix
        return matrix

    def portfolio_loss(self, W, cell_return, cell_risk):
        """
        Kahneman-Tversky asymmetric portfolio loss.
        Components:
          1. -Sharpe: maximize Sharpe ratio (sign inverted).
          2. Loss aversion: 2.5x penalty on risk (Prospect Theory).
          3. Concentration penalty: prevent over-concentration in one cell.
          4. Entropy: encourage diversification.
          5. Marginal: row/col sum constraints (Sinkhorn complement).
        """
        pr = np.sum(W * cell_return)
        pk = max(np.sum(W * cell_risk), 0.001)
        sharpe = pr / pk

        Ws = np.maximum(W, 1e-10)

        L_sharpe = -LAMBDA_SHARPE * sharpe
        L_risk = LOSS_AVERSION * LAMBDA_RISK * np.sum(W * cell_risk)
        L_conc = LAMBDA_CONCENTRATION * np.sum(np.maximum(W - MAX_CELL_ALLOCATION, 0)**2)
        L_ent = LAMBDA_ENTROPY * np.sum(Ws * np.log(Ws))
        L_marg = LAMBDA_MARGINAL * (
            np.sum((W.sum(1) - np.array(TIME_MARGINALS))**2) +
            np.sum((W.sum(0) - np.array(RISK_MARGINALS))**2)
        )

        total = L_sharpe + L_risk + L_conc + L_ent + L_marg
        return total, sharpe, pr, pk

    def train_step(self, X, cell_return, cell_risk, row_m, col_m, lr):
        """One step: forward -> loss -> numerical gradient on logits -> analytical backprop."""
        W = self.forward(X, row_m, col_m)
        loss, sharpe, pr, pk = self.portfolio_loss(W, cell_return, cell_risk)

        # Numerical gradient on 9 logits
        logits = self.cache['logits']
        dlogits = np.zeros(MATRIX_OUTPUT)
        eps = 1e-4
        for k in range(MATRIX_OUTPUT):
            lp = logits.copy(); lp[k] += eps
            lm = logits.copy(); lm[k] -= eps
            Lp, _, _, _ = self.portfolio_loss(sinkhorn_3x3(lp, row_m, col_m), cell_return, cell_risk)
            Lm, _, _, _ = self.portfolio_loss(sinkhorn_3x3(lm, row_m, col_m), cell_return, cell_risk)
            dlogits[k] = (Lp - Lm) / (2 * eps)

        # Analytical backprop through hidden layers
        dout = dlogits.reshape(1, -1)
        self.t_adam += 1
        acts = self.cache['activations']

        for i in reversed(range(len(self.layers))):
            dW = acts[i].T @ dout
            db = dout.copy()

            if i > 0:
                dout = (dout @ self.layers[i]['W'].T) * (acts[i] > 0).astype(float)

            # Adam update
            L = self.layers[i]
            if L['mW'] is None:
                L['mW'] = np.zeros_like(dW); L['vW'] = np.zeros_like(dW)
                L['mb'] = np.zeros_like(db); L['vb'] = np.zeros_like(db)

            L['mW'] = ADAM_BETA1 * L['mW'] + (1-ADAM_BETA1) * dW
            L['vW'] = ADAM_BETA2 * L['vW'] + (1-ADAM_BETA2) * dW**2
            L['mb'] = ADAM_BETA1 * L['mb'] + (1-ADAM_BETA1) * db
            L['vb'] = ADAM_BETA2 * L['vb'] + (1-ADAM_BETA2) * db**2

            t = self.t_adam
            L['W'] -= lr * (L['mW']/(1-ADAM_BETA1**t)) / (np.sqrt(L['vW']/(1-ADAM_BETA2**t)) + ADAM_EPSILON)
            L['b'] -= lr * (L['mb']/(1-ADAM_BETA1**t)) / (np.sqrt(L['vb']/(1-ADAM_BETA2**t)) + ADAM_EPSILON)

        return loss, sharpe, pr, pk, W

    @property
    def n_params(self):
        return sum(l['W'].size + l['b'].size for l in self.layers)
