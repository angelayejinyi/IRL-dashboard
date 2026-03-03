"""
MaxEnt IRL algorithm with value iteration and step-by-step outputs.
Usage: outputs = maxent_irl_onehot(P, trajectories, nS=5, nA=3, epochs=50)
       each IRLOutput in list contains alpha, reward, policy, Q, V, mu_E, mu_pi, epoch
"""

from __future__ import annotations

from typing import Optional
import numpy as np


# ==================== Helpers ====================


def sa_to_index(s: int, a_idx: int, nA: int) -> int:
    """Flatten (state, action_index) -> row index in [0, nS*nA)."""
    return s * nA + a_idx


def index_to_sa(idx: int, nA: int) -> tuple[int, int]:
    """Inverse of sa_to_index."""
    return idx // nA, idx % nA


def logsumexp(x: np.ndarray) -> float:
    """Stable logsumexp for 1D array."""
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))


def softmax_rowwise(Q: np.ndarray) -> np.ndarray:
    """Row-wise softmax over actions. Q shape: (nS, nA) -> policy (nS, nA)."""
    Q_stable = Q - Q.max(axis=1, keepdims=True)
    expQ = np.exp(Q_stable)
    return expQ / expQ.sum(axis=1, keepdims=True)


# ==================== Value Iteration ====================


def value_iteration_Q(
    P: np.ndarray,
    r: np.ndarray,
    gamma: float,
    threshold: float = 1e-4,
    max_iter: int = 10_000,
    history: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list]:
    """
    Compute optimal Q via value iteration. Initialize Q=0.
    V(s) = max_a Q(s,a)
    Q_new(s,a) = r(s,a) + gamma * sum_s' P[s,a,s'] * V(s')
    
    If history=True, also returns list of Q snapshots per iteration.
    """
    nS, nA, _ = P.shape
    Q = np.zeros((nS, nA), dtype=float)
    Q_history = [Q.copy()] if history else None

    diff = float("inf")
    it = 0
    while diff > threshold and it < max_iter:
        it += 1
        V = np.max(Q, axis=1)
        Q_new = np.zeros_like(Q)

        diff = 0.0
        for s in range(nS):
            for a in range(nA):
                Q_sa = r[s, a] + gamma * float(np.dot(P[s, a, :], V))
                Q_new[s, a] = Q_sa
                diff = max(diff, abs(Q_new[s, a] - Q[s, a]))

        Q = Q_new
        if history:
            Q_history.append(Q.copy())

    V = np.max(Q, axis=1)
    if history:
        return Q, V, Q_history
    return Q, V


# ==================== Expert / Occupancy ====================


def feature_matrix_sa_onehot(nS: int, nA: int) -> np.ndarray:
    """One-hot features for each (s,a). F shape: (nS*nA, nS*nA) = I."""
    return np.eye(nS * nA)


def expert_feature_expectations_onehot(
    trajectories: list[list[tuple[int, int]]],
    nS: int,
    nA: int,
    normalize_by_length: bool = True,
) -> np.ndarray:
    """Returns μ_E in R^{nS*nA} for one-hot (s,a) features."""
    mu = np.zeros(nS * nA, dtype=float)
    total_steps = 0
    for traj in trajectories:
        for s, a in traj:
            a_idx = int(a) + 1  # -1->0, 0->1, 1->2
            mu[sa_to_index(int(s), a_idx, nA)] += 1.0
            total_steps += 1

    if normalize_by_length and total_steps > 0:
        mu /= total_steps

    return mu


def start_state_dist_from_trajectories(
    trajectories: list[list[tuple[int, int]]],
    nS: int,
) -> np.ndarray:
    """Empirical p0(s) from trajectories' first states."""
    p0 = np.zeros(nS, dtype=float)
    if len(trajectories) == 0:
        p0[:] = 1.0 / nS
        return p0

    for traj in trajectories:
        s0, _ = traj[0]
        p0[int(s0)] += 1.0
    p0 /= p0.sum()
    return p0


def expected_occupancy_sa(
    P: np.ndarray,
    policy: np.ndarray,
    p0: np.ndarray,
    horizon: int,
    normalize_by_length: bool = True,
) -> np.ndarray:
    """Computes ρ_π(s,a) as expected (s,a) occupancy. Returns vector in R^{nS*nA}."""
    nS, nA, _ = P.shape
    d = p0.copy()
    rho_sa = np.zeros((nS, nA), dtype=float)

    for _t in range(horizon):
        rho_t = d[:, None] * policy  # (nS, nA)
        rho_sa += rho_t

        d_next = np.zeros(nS, dtype=float)
        for s in range(nS):
            for a in range(nA):
                d_next += rho_t[s, a] * P[s, a, :]
        d = d_next

    if normalize_by_length and horizon > 0:
        rho_sa /= horizon

    return rho_sa.reshape(-1)


# ==================== IRL Output Container ====================


class IRLOutput:
    """Container for one epoch of MaxEnt IRL outputs."""

    def __init__(
        self,
        alpha: np.ndarray,
        reward: np.ndarray,
        policy: np.ndarray,
        Q: np.ndarray,
        V: np.ndarray,
        mu_E: np.ndarray,
        mu_pi: np.ndarray,
        epoch: int = 0,
        gradient_norm: float = 0.0,
    ):
        self.alpha = alpha
        self.reward = reward
        self.policy = policy
        self.Q = Q
        self.V = V
        self.mu_E = mu_E
        self.mu_pi = mu_pi
        self.epoch = epoch
        self.gradient_norm = gradient_norm


# ==================== MaxEnt IRL Main Loop ====================


def maxent_irl_onehot(
    P: np.ndarray,
    trajectories: list[list[tuple[int, int]]],
    nS: int,
    nA: int,
    gamma: float = 0.9,
    epochs: int = 50,
    learning_rate: float = 0.1,
    vi_threshold: float = 1e-4,
    horizon: Optional[int] = None,
    seed: int = 0,
) -> list[IRLOutput]:
    """
    MaxEnt IRL loop returning per-epoch outputs.
    Returns list of IRLOutput objects, one per epoch.
    """
    F = feature_matrix_sa_onehot(nS, nA)
    d_features = F.shape[1]

    mu_E = expert_feature_expectations_onehot(trajectories, nS=nS, nA=nA, normalize_by_length=True)

    p0 = start_state_dist_from_trajectories(trajectories, nS=nS)
    if horizon is None:
        lens = [len(t) for t in trajectories if len(t) > 0]
        horizon = int(np.median(lens)) if lens else 10
        horizon = max(1, horizon)

    np.random.seed(seed)
    alpha = np.random.uniform(size=(d_features,))

    outputs = []
    for ep in range(epochs):
        r_vec = F @ alpha
        r = r_vec.reshape(nS, nA)

        Q, V = value_iteration_Q(P, r, gamma=gamma, threshold=vi_threshold)

        policy = softmax_rowwise(Q)

        mu_pi = expected_occupancy_sa(P, policy, p0, horizon=horizon, normalize_by_length=True)

        grad = mu_E - mu_pi
        grad_norm = float(np.linalg.norm(grad))
        alpha = alpha + learning_rate * grad

        out = IRLOutput(
            alpha=alpha.copy(),
            reward=r.copy(),
            policy=policy.copy(),
            Q=Q.copy(),
            V=V.copy(),
            mu_E=mu_E.copy(),
            mu_pi=mu_pi.copy(),
            epoch=ep,
            gradient_norm=grad_norm,
        )
        outputs.append(out)

    return outputs
