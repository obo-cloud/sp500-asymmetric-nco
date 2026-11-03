"""
Canonical Asymmetric Downside-Risk Nested Clustered Optimization (Asym-Sticky-NCO)
library for S&P 500 sector allocation.

This module is the single source of truth for the paper:

  "Mitigating Return Drag and Regime Whipsaws in Sector Rotation:
   An Asymmetric Downside-Aware Nested Clustered Optimization Framework"

Implemented components:
1. Bawa-Lindenberg downside semi-covariance matrix (target = monthly risk-free rate).
2. Downside semi-correlation distance with Ward hierarchical clustering (K = 3).
3. Two-stage Nested Clustered Optimization (intra-cluster + inter-cluster).
4. Sticky 2-state Gaussian HMM with self-transition prior (kappa = 0.30) and
   confidence-adaptive threshold (delta = 0.65).
5. Dual-regime objectives: Sortino utility (gamma = 1.2, Bull) and minimum
   downside variance (Bear), both with L1 turnover regularization.
"""

import contextlib
import io

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

LOOKBACK = 24
N_CLUSTERS = 3
KAPPA = 0.30
DELTA = 0.65
GAMMA = 1.2
LAMBDA_TURN_PROPOSED = 0.015
LAMBDA_TURN_BASELINE = 0.01
EPS = 1e-6
SEED = 42


def compute_downside_semicov(returns_matrix, target=0.0):
    """
    Bawa-Lindenberg downside semi-covariance:
        Sigma^-_ij = (1 / T) * sum_t min(R_it - tau_t, 0) * min(R_jt - tau_t, 0)
    with spectral ridge regularization to guarantee positive semi-definiteness.
    """
    T, k = returns_matrix.shape
    D = np.minimum(returns_matrix - target, 0.0)
    semicov = np.dot(D.T, D) / max(T, 1)
    semicov = (semicov + semicov.T) / 2.0
    min_eig = np.min(np.linalg.eigvalsh(semicov))
    if min_eig < 1e-8:
        semicov += (abs(min_eig) + EPS) * np.eye(k)
    return semicov


def optimize_portfolio(mu, cov, obj_type='min_variance', rf=0.0,
                       w_prev=None, lambda_turn=0.0):
    """
    Quadratic optimizer over the unit simplex.
    obj_type:
      'min_variance': min w' Cov w
      'max_sharpe'  : min 0.5 * w' Cov w - 1.0 * (w' mu - rf)
      'max_sortino' : min 0.5 * w' Cov w - GAMMA * (w' mu - rf)
    Optional L1 turnover regularization lambda_turn * ||w - w_prev||_1.
    """
    k = cov.shape[0]
    w0 = np.ones(k) / k if w_prev is None else w_prev.copy()
    bounds = [(0.0, 1.0) for _ in range(k)]
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    def obj_func(w):
        var = np.dot(w.T, np.dot(cov, w))
        if obj_type == 'max_sharpe':
            ret = np.dot(w, mu) - rf
            cost = 0.5 * var - 1.0 * ret
        elif obj_type == 'max_sortino':
            ret = np.dot(w, mu) - rf
            cost = 0.5 * var - GAMMA * ret
        else:
            cost = var
        if w_prev is not None and lambda_turn > 0:
            cost += lambda_turn * np.sum(np.abs(w - w_prev))
        return cost

    res = minimize(obj_func, w0, method='SLSQP', bounds=bounds,
                   constraints=constraints)
    if res.success:
        w = np.maximum(res.x, 0.0)
        return w / np.sum(w)
    return w0


def run_nco(returns_subset, use_downside=False, objective='min_variance',
            rf=0.0, w_prev=None, lambda_turn=0.0, n_clusters=N_CLUSTERS):
    """
    Nested Clustered Optimization:
    1. Cluster assets via Ward linkage on the (downside or symmetric) distance.
    2. Optimize intra-cluster weights with turnover regularization.
    3. Optimize inter-cluster weights over the collapsed cluster returns.
    4. Recombine w = w_intra * w_inter (renormalized).
    """
    k = returns_subset.shape[1]
    ret_vals = returns_subset.values

    if use_downside:
        cov = compute_downside_semicov(ret_vals, target=rf)
        diag = np.sqrt(np.diag(cov))
        outer = np.outer(diag, diag)
        outer = np.where(outer == 0, 1e-12, outer)
        corr = np.clip(cov / outer, -1.0, 1.0)
    else:
        cov = np.cov(ret_vals, rowvar=False)
        corr = np.corrcoef(ret_vals, rowvar=False)

    dist = np.sqrt(0.5 * np.maximum(1.0 - corr, 0.0))
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='ward')
    cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    unique_clusters = np.unique(cluster_labels)

    w_intra = np.zeros(k)
    for c in unique_clusters:
        idx = np.where(cluster_labels == c)[0]
        cov_c = cov[np.ix_(idx, idx)]
        mu_c = np.mean(ret_vals[:, idx], axis=0)
        w_prev_c = w_prev[idx] if w_prev is not None else None
        w_intra[idx] = optimize_portfolio(mu_c, cov_c, obj_type=objective,
                                          rf=rf, w_prev=w_prev_c,
                                          lambda_turn=lambda_turn)

    cluster_returns_list = []
    for c in unique_clusters:
        idx = np.where(cluster_labels == c)[0]
        r_c = np.dot(ret_vals[:, idx], w_intra[idx])
        cluster_returns_list.append(r_c)
    cluster_returns = np.column_stack(cluster_returns_list)

    if use_downside:
        cov_inter = compute_downside_semicov(cluster_returns, target=rf)
    else:
        cov_inter = np.cov(cluster_returns, rowvar=False)

    mu_inter = np.mean(cluster_returns, axis=0)
    w_inter = optimize_portfolio(mu_inter, cov_inter, obj_type=objective,
                                 rf=rf, lambda_turn=0.0)

    w_final = np.zeros(k)
    for i, c in enumerate(unique_clusters):
        idx = np.where(cluster_labels == c)[0]
        w_final[idx] = w_intra[idx] * w_inter[i]

    w_final = np.maximum(w_final, 0.0)
    return w_final / np.sum(w_final)


def fit_hmm(observations, sticky_kappa=0.0, covariance_type='diag',
            n_iter=60, tol=1e-3):
    """
    Fit a 2-state Gaussian HMM with optional sticky diagonal self-transition
    prior. Informational Baum-Welch non-convergence messages from hmmlearn
    are suppressed; estimation errors propagate as exceptions.
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        model = GaussianHMM(n_components=2, covariance_type=covariance_type,
                            n_iter=n_iter, tol=tol, random_state=SEED)
        model.fit(observations)

    if sticky_kappa > 0:
        P = model.transmat_.copy()
        for i in range(model.n_components):
            P[i, i] += sticky_kappa
            P[i, :] /= np.sum(P[i, :])
        model.transmat_ = P
    return model


def detect_bull_id(model):
    """Bull state = the state with the higher mean on feature column 0."""
    return int(np.argmax(model.means_[:, 0]))


def apply_confidence_threshold(last_is_bull, p_bull, delta=DELTA):
    """
    Confidence-adaptive regime update:
    switch only when the alternative state posterior crosses delta.
    """
    if last_is_bull:
        return 0 if (1.0 - p_bull) >= delta else 1
    return 1 if p_bull >= delta else 0
