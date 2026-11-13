"""Kurtosis-adaptive constellation family for OFDM-ISAC.

Python implementation of the pretrained kurtosis-adaptive constellation
family (M = 64 points) and of the functions used to evaluate it:

- ``generate_adaptive_constellation``: forward pass of the neural networks,
  Eqs. (12)-(14) of [1]
- ``mutual_information``: MI over AWGN (Gauss-Hermite quadrature)
- ``mi_upper_bound_kurtosis``: upper bound on the MI under a kurtosis
  constraint, Eq. (11) of [1] and Eqs. (23)-(32) of [2]

This is a one-to-one port of the MATLAB implementation in ``../matlab/`` and
uses the same weight file ``../matlab/adaptive_constellation_weights.mat``.

References
----------
[1] B. Geiger, L. Hampel, and L. Schmalen, "Joint power and
    adaptive-constellation optimization for OFDM-ISAC systems," in Proc.
    IEEE Conf. Integr. Sens. Commun. (ISAC), Lisbon, Portugal, Nov. 2026.
    arXiv:2611.12166
[2] B. Geiger et al., "Constellation shaping for OFDM-ISAC systems: From
    theoretical bounds to practical implementation," IEEE Trans. Commun.,
    2026. arXiv:2509.04055

License: MIT (see LICENSE in repository root)
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import brentq
from scipy.special import erfcx

WEIGHTS_FILE = (Path(__file__).resolve().parent.parent
                / "matlab" / "adaptive_constellation_weights.mat")


def load_weights(path=WEIGHTS_FILE):
    """Load the trained NN weights.

    Returns a dict with the constellation size ``M``, the constants
    ``eps_R`` and ``eps_Pr`` of Eqs. (12) and (14), and the layers of the
    radius, phase, and probability networks as lists of ``(W, b)`` tuples
    with ``W`` of shape [out x in] and ``b`` of shape [out].
    """
    mat = loadmat(path, squeeze_me=False, struct_as_record=False)

    def layers(nn):
        num_layers = len(nn._fieldnames) // 2
        return [(np.asarray(getattr(nn, f"W{l}"), dtype=float),
                 np.asarray(getattr(nn, f"b{l}"), dtype=float).ravel())
                for l in range(1, num_layers + 1)]

    return {
        "M": int(mat["M"].item()),
        "eps_R": float(mat["eps_R"].item()),
        "eps_Pr": float(mat["eps_Pr"].item()),
        "radial_nn": layers(mat["radial_nn"][0, 0]),
        "phase_nn": layers(mat["phase_nn"][0, 0]),
        "prob_nn": layers(mat["prob_nn"][0, 0]),
    }


def _mlp_forward(inp, layers):
    """Fully connected NN with ReLU activations in the hidden layers and a
    linear output layer. ``inp`` has shape [K x 1]."""
    out = inp
    for l, (W, b) in enumerate(layers):
        out = out @ W.T + b
        if l < len(layers) - 1:
            out = np.maximum(out, 0.0)
    return out


def generate_adaptive_constellation(kappa_target, weights=None):
    """Kurtosis-adaptive constellation, Sec. IV and Eqs. (12)-(14) of [1].

    Parameters
    ----------
    kappa_target : float or array_like
        Target kurtosis, 1 <= kappa_target <= 2 (1: unit modulus /
        sensing-optimal, 2: Gaussian-like / communication-optimal).
        An array of K values generates K constellations at once.
    weights : dict, optional
        Output of :func:`load_weights`; pass it when calling the function
        many times to avoid reloading the file.

    Returns
    -------
    x : ndarray, complex
        Constellation points with zero mean and unit average power,
        shape [M] for a scalar ``kappa_target`` and [K x M] otherwise.
    p : ndarray
        Probabilities of the constellation points (summing to one along the
        last axis), same shape as ``x``.

    The achieved kurtosis is ``np.sum(p * np.abs(x)**4, axis=-1)``.
    """
    if weights is None:
        weights = load_weights()
    kappa = np.asarray(kappa_target, dtype=float)
    scalar_input = kappa.ndim == 0
    kappa = kappa.reshape(-1, 1)                                   # [K x 1]
    if np.any((kappa < 1) | (kappa > 2)):
        raise ValueError("kappa_target must be in [1, 2].")
    M = weights["M"]

    # Radii, Eq. (12): unit modulus for kappa = 1, radial shaping for kappa > 1
    radial_out = _mlp_forward(kappa, weights["radial_nn"])         # [K x M]
    r = np.maximum(1 + np.sqrt(kappa - 1) * np.tanh(radial_out),
                   weights["eps_R"])

    # Phases, Eq. (13): bounded offset added to a uniform angular grid
    phase_out = _mlp_forward(kappa, weights["phase_nn"])           # [K x M]
    theta_grid = 2 * np.pi * np.arange(M) / M
    theta = theta_grid + np.pi * np.tanh(phase_out)

    # Probabilities, Eq. (14): uniform for kappa = 1, shaped for kappa > 1
    logits = _mlp_forward(kappa, weights["prob_nn"])               # [K x M]
    logits = np.minimum((kappa - 1) / weights["eps_Pr"], 1) * logits
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)

    # Constellation points, normalized to zero mean and unit average power
    x = r * np.exp(1j * theta)
    x = x - np.sum(p * x, axis=1, keepdims=True)
    x = x / np.sqrt(np.sum(p * np.abs(x) ** 2, axis=1, keepdims=True))

    if scalar_input:
        return x[0], p[0]
    return x, p


def mutual_information(x, p, snr_dB, num_nodes=32):
    """Mutual information (bit/symbol) of the constellation ``x`` with
    probabilities ``p`` over the AWGN channel y = x + w, w ~ CN(0, sigma^2),
    with SNR = 1/sigma^2 (``x`` has unit average power):

        I(X;Y) = -sum_i p_i E_w{ log2( sum_j p_j
                    exp((|w|^2 - |x_i - x_j + w|^2) / sigma^2) ) }.

    The expectation over the complex Gaussian noise w is evaluated with a
    two-dimensional Gauss-Hermite quadrature (``num_nodes`` x ``num_nodes``
    nodes). In contrast to a Monte Carlo estimate, the result is
    deterministic and free of sampling noise.
    """
    x = np.asarray(x).ravel()
    p = np.asarray(p, dtype=float).ravel()
    p = p / p.sum()
    sigma2 = 10 ** (-snr_dB / 10)

    # Quadrature points w = sigma*(t_k + 1j*t_l) of CN(0, sigma^2), stored as
    # [Re{w}, Im{w}], with weights g_k*g_l/pi (summing to one)
    t, g = np.polynomial.hermite.hermgauss(num_nodes)
    t_re, t_im = np.meshgrid(t, t, indexing="ij")
    w = np.sqrt(sigma2) * np.stack([t_re.ravel(), t_im.ravel()], axis=1)  # [K^2 x 2]
    q = np.outer(g, g).ravel() / np.pi                                    # [K^2]

    with np.errstate(divide="ignore"):
        log_p = np.log(p)
    mi = 0.0
    for i in np.flatnonzero(p > 0):
        # Exponents (|w|^2 - |d_ij + w|^2)/sigma^2 + log(p_j) with
        # d_ij = x_i - x_j, using |w|^2 - |d + w|^2 = -|d|^2 - 2 Re{d^* w}
        d = x[i] - x
        e = (-(np.abs(d) ** 2) - 2 * (w @ np.stack([d.real, d.imag]))) / sigma2 + log_p
        e_max = e.max(axis=1, keepdims=True)                              # [K^2 x 1]
        log_sum = e_max[:, 0] + np.log(np.exp(e - e_max).sum(axis=1))     # log-sum-exp
        mi -= p[i] * (q @ log_sum)
    return mi / np.log(2)


def mi_upper_bound_kurtosis(kappa, snr_dB):
    """Upper bound on the MI under a kurtosis constraint.

    Evaluates the upper bound on the mutual information (bit/symbol) of the
    complex AWGN channel y = x + w, w ~ CN(0, sigma^2), for a unit-power
    input x with kurtosis E{|x|^4} = kappa, see Eqs. (8)-(11) of [1] with
    h = 1 and P = 1, and Eqs. (23)-(32) of [2].

    The bound is obtained from the maximum-entropy output PDF
    exp(gamma0 + gamma2*|y|^2 + gamma4*|y|^4) that satisfies the second- and
    fourth-order moment constraints of the received signal.

    Parameters
    ----------
    kappa : float or array_like
        Kurtosis values, 1 <= kappa <= 2.
    snr_dB : float
        Communication SNR 1/sigma^2 in dB.
    """
    kappa = np.asarray(kappa, dtype=float)
    if np.any((kappa < 1) | (kappa > 2)):
        raise ValueError("kappa must be in [1, 2].")
    sigma2 = 10 ** (-snr_dB / 10)
    entropy_awgn = np.log2(np.pi * np.e * sigma2)

    # Moment constraints of the received signal, Eqs. (8) and (9)
    C0 = 1.0
    C2 = 1 + sigma2
    C4 = kappa + 4 * sigma2 + 2 * sigma2 ** 2

    mi_ub = np.zeros(kappa.shape)
    for i in np.ndindex(kappa.shape):
        if kappa[i] >= 2 - 1e-9:
            # Gaussian input: maximum-entropy output is Gaussian -> capacity
            mi_ub[i] = np.log2(1 + 1 / sigma2)
            continue
        c4 = C4[i]

        # Nonlinear equation in gamma2 (Eq. (29) in [2]);
        # erfcx(z) = exp(z^2)*erfc(z) is used for numerical stability
        def f(g2):
            a = g2 * C2 + 1
            return ((C2 * a / c4 - g2) * np.sqrt(np.pi * c4 / (2 * a))
                    * erfcx(-g2 * np.sqrt(c4 / (2 * a))) - C0)

        # Bracket the root. gamma2 > -1/C2 keeps the square root real; f
        # tends to zero at this boundary (trivial solution), so the bracket
        # is found by scanning logarithmically spaced offsets from the
        # boundary for the first sign change of f from positive to negative.
        g2_hi = 10.0
        while f(g2_hi) > 0:
            g2_hi *= 2
        g2_grid = -1 / C2 + np.logspace(-8, np.log10(g2_hi + 1 / C2), 400)
        f_grid = f(g2_grid)
        pos = np.flatnonzero(f_grid > 1e-14)              # skip round-off region
        if pos.size == 0 or not np.any(f_grid[pos[0]:] < 0):
            raise RuntimeError(f"No valid bracket found for kappa = {kappa[i]}.")
        idx = pos[0] + np.flatnonzero(f_grid[pos[0]:] < 0)[0]
        gamma2 = brentq(f, g2_grid[idx - 1], g2_grid[idx], xtol=1e-14)

        # gamma0 and gamma4 (Eqs. (30) and (31) in [2])
        gamma0 = np.log(1 / np.pi * (C2 * (gamma2 * C2 + 1) / c4 - gamma2))
        gamma4 = -(gamma2 * C2 + 1) / (2 * c4)

        # Differential entropy of the output (Eq. (32) in [2]) and bound,
        # Eq. (11) of [1]
        entropy_out = -(gamma0 + C2 * gamma2 + c4 * gamma4) / np.log(2)
        mi_ub[i] = entropy_out - entropy_awgn

    return mi_ub.item() if mi_ub.ndim == 0 else mi_ub
