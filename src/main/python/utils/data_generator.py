from typing import Tuple

import numpy as np

def generate_test_data(n_assets=100, seed=42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate synthetic portfolio data.

    Args:
        n_assets: Number of assets
        seed: Random seed

    Returns:
        Tuple of (returns, risks, correlation matrix, risk capacity)
    """
    np.random.seed(seed)

    # Base parameters
    returns = np.abs(np.random.normal(0.1, 0.05, n_assets)).astype(np.float32)
    risks = np.abs(np.random.normal(0.15, 0.05, n_assets)).astype(np.float32)

    # Correlation matrix (ensure symmetry and positive-definiteness)
    corr = np.eye(n_assets, dtype=np.float32)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            corr[i,j] = corr[j,i] = np.clip(np.random.normal(0.3, 0.2), -0.5, 0.9)

    # Ensure positive definiteness
    min_eig = np.min(np.linalg.eigvals(corr))
    if min_eig < 0:
        corr += (-min_eig + 1e-6) * np.eye(n_assets)
        # Renormalize
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)

    # Risk capacity (20% of total risk)
    C = 0.2 * np.sum(risks)

    return returns, risks, corr, float(C)
