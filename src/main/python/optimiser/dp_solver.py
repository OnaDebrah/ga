from typing import List, Tuple, Any

import numpy as np
from numpy._typing import _64Bit

# class KnapsackSolver:
#     """Class for solving knapsack problems using dynamic programming."""
#
#     @staticmethod
#     def solve(values: np.ndarray, weights: np.ndarray, capacity: float, precision: int = 100) -> tuple[
#         np.ndarray[Any, np.dtype[np.floating[_64Bit]]], Any]:
#         """
#         Solve knapsack problem using dynamic programming.
#
#         Args:
#             values: Array of item values
#             weights: Array of item weights
#             capacity: Maximum capacity
#             precision: Precision factor for discretizing capacity
#
#         Returns:
#             Tuple containing optimal value and selected items
#         """
#         n = len(values)
#         cap_discrete = int(capacity * precision) + 1
#
#         dp = np.zeros(cap_discrete)
#         selected = np.zeros((n, cap_discrete), dtype=np.int8)
#
#         for i in range(n):
#             weight_i = int(weights[i] * precision)
#             value_i = values[i]
#             for cap in range(cap_discrete - 1, weight_i - 1, -1):
#                 if dp[cap - weight_i] + value_i > dp[cap]:
#                     dp[cap] = dp[cap - weight_i] + value_i
#                     selected[i, cap] = 1
#
#         solution = np.zeros(n, dtype=np.int8)
#         cap = cap_discrete - 1
#         for i in range(n - 1, -1, -1):
#             weight_i = int(weights[i] * precision)
#             if cap >= weight_i and selected[i, cap] == 1:
#                 solution[i] = 1
#                 cap -= weight_i
#
#         return dp[cap_discrete - 1], solution.tolist()

class KnapsackSolver:
    """Class for solving knapsack problems using dynamic programming."""

    @staticmethod
    def solve(values: np.ndarray, weights: np.ndarray, capacity: float, precision: int = 100) -> tuple[
        np.ndarray[Any, np.dtype[np.floating[_64Bit]]], Any]:
        """
        Solve knapsack problem using dynamic programming.

        Args:
            values: Array of item values
            weights: Array of item weights
            capacity: Maximum capacity
            precision: Precision factor for discretizing capacity

        Returns:
            Tuple containing optimal value and selected items
        """
        n = len(values)
        # Discretize capacity to handle floating point weights
        cap_discrete = int(capacity * precision) + 1

        # Initialize DP table and tracking
        dp = np.zeros(cap_discrete)
        selected = np.zeros((n, cap_discrete), dtype=np.int8)

        # DP algorithm
        for i in range(n):
            w = int(weights[i] * precision)
            for j in range(cap_discrete - 1, w - 1, -1):
                if dp[j - w] + values[i] > dp[j]:
                    dp[j] = dp[j - w] + values[i]
                    selected[i, j] = 1

        # Reconstruct solution
        solution = np.zeros(n, dtype=np.int8)
        j = cap_discrete - 1
        for i in range(n - 1, -1, -1):
            w = int(weights[i] * precision)
            if j >= w and selected[i, j] == 1:
                solution[i] = 1
                j -= w

        return dp[cap_discrete - 1], solution